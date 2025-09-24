#!/usr/bin/env python3
import argparse, onnx
from onnx import helper, numpy_helper, TensorProto, shape_inference

def np_array(x, dtype):
    import numpy as np
    return np.array(x, dtype=dtype)

def get_shape(model, name):
    g = model.graph
    for space in (g.value_info, g.output, g.input):
        for v in space:
            if v.name == name and v.type.HasField("tensor_type"):
                dims=[]
                for d in v.type.tensor_type.shape.dim:
                    dims.append(int(d.dim_value) if d.HasField("dim_value") else None)
                return dims
    return None

def current_opset(model):
    for oi in model.opset_import:
        if oi.domain in ("","ai.onnx"):
            return oi.version
    return 13

def add_reducesum_clip(model, class_tensor, H, W, opset):
    g = model.graph
    rs_out  = class_tensor + "_sum"
    clip_out= class_tensor + "_sum_clip"
    # ReduceSum: для opset>=13 ось передаётся входом
    if opset >= 13:
        axes_name = class_tensor + "_axes"
        g.initializer.extend([numpy_helper.from_array(np_array([1], "int64"), axes_name)])
        rs = helper.make_node("ReduceSum",
                              inputs=[class_tensor, axes_name], outputs=[rs_out],
                              name=class_tensor + "_ReduceSum", keepdims=1)
    else:
        rs = helper.make_node("ReduceSum",
                              inputs=[class_tensor], outputs=[rs_out],
                              name=class_tensor + "_ReduceSum", keepdims=1, axes=[1])

    min_name = class_tensor + "_clip_min"
    max_name = class_tensor + "_clip_max"
    g.initializer.extend([
        numpy_helper.from_array(np_array([0.0], "float32"), min_name),
        numpy_helper.from_array(np_array([1.0], "float32"), max_name),
    ])
    clip = helper.make_node("Clip",
                            inputs=[rs_out, min_name, max_name], outputs=[clip_out],
                            name=class_tensor + "_Clip")
    g.node.extend([rs, clip])
    # объявим форму [1,1,H,W]
    g.value_info.extend([helper.make_tensor_value_info(clip_out, TensorProto.FLOAT, [1,1,H,W])])
    return clip_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--bbox", required=True, help="три имени выходов Conv (bbox) P3,P4,P5 через запятую")
    ap.add_argument("--cls",  required=True, help="три имени выходов Conv (cls)  P3,P4,P5 через запятую")
    args = ap.parse_args()

    model = onnx.load(args.inp)
    model = shape_inference.infer_shapes(model, strict_mode=False)
    opset = current_opset(model)

    bbox_names = [s.strip() for s in args.bbox.split(",")]
    cls_names  = [s.strip() for s in args.cls.split(",")]
    assert len(bbox_names)==3 and len(cls_names)==3, "Нужно по 3 имени для --bbox и --cls"

    g = model.graph
    del g.output[:]  # переопределяем порядок выходов

    for b,c in zip(bbox_names, cls_names):
        shp_b = get_shape(model, b)  # [1,4,H,W]
        shp_c = get_shape(model, c)  # [1,C,H,W]
        assert shp_b and shp_c and len(shp_b)==4 and len(shp_c)==4, f"Нет формы у {b} или {c}"
        _,_,H,W = shp_b
        _,C,_,_ = shp_c

        # 1) bbox
        g.output.extend([helper.make_tensor_value_info(b, TensorProto.FLOAT, [1,4,H,W])])
        # 2) class
        g.output.extend([helper.make_tensor_value_info(c, TensorProto.FLOAT, [1,C,H,W])])
        # 3) sum = ReduceSum+Clip
        s = add_reducesum_clip(model, c, H, W, opset)
        g.output.extend([helper.make_tensor_value_info(s, TensorProto.FLOAT, [1,1,H,W])])

    model = shape_inference.infer_shapes(model, strict_mode=False)
    onnx.checker.check_model(model)
    onnx.save(model, args.outp)
    print("Saved:", args.outp)

if __name__ == "__main__":
    main()
