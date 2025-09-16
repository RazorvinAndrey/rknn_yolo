#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import onnx
from io import BytesIO

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint

# ---- helper: find head lists (reg_preds / cls_preds) ----
def find_head_lists(model):
    for m in model.modules():
        if hasattr(m, 'reg_preds') and hasattr(m, 'cls_preds'):
            reg_preds = getattr(m, 'reg_preds')
            cls_preds = getattr(m, 'cls_preds')
            if isinstance(reg_preds, (nn.ModuleList, list)) and isinstance(cls_preds, (nn.ModuleList, list)):
                nl = len(reg_preds)
                # try to get number of classes
                for k in ('num_classes', 'nc', 'num_cls'):
                    if hasattr(m, k):
                        nc = int(getattr(m, k))
                        break
                else:
                    # fallback to model-level
                    nc = None
                    for k in ('num_classes', 'nc', 'num_cls'):
                        if hasattr(model, k):
                            nc = int(getattr(model, k)); break
                return m, reg_preds, cls_preds, nl, nc
    return None, None, None, None, None

# ---- wrapper that exposes head conv outputs (RKNN-optimized style) ----
class OnnxExportWrapper(nn.Module):
    def __init__(self, model, reg_preds, cls_preds, nl):
        super().__init__()
        self.model = model
        self.reg_preds = reg_preds
        self.cls_preds = cls_preds
        self.nl = nl
        self._reg_out = [None] * nl
        self._cls_out = [None] * nl
        self._hooks = []
        # attach hooks to conv *outputs* we want to export
        for i in range(nl):
            self._hooks.append(self.reg_preds[i].register_forward_hook(self._make_hook(self._reg_out, i)))
            self._hooks.append(self.cls_preds[i].register_forward_hook(self._make_hook(self._cls_out, i)))

    @staticmethod
    def _make_hook(storage, idx):
        def hook(module, inp, out):
            storage[idx] = out
        return hook

    def forward(self, x):
        # reset buffers
        for i in range(self.nl):
            self._reg_out[i] = None
            self._cls_out[i] = None
        # run full model; hooks will capture tensors we need
        _ = self.model(x)

        outs = []
        for i in range(self.nl):
            reg = self._reg_out[i]             # (B,4,H,W)
            cls_logits = self._cls_out[i]      # (B,C,H,W)
            cls = torch.sigmoid(cls_logits)    # sigmoid, как в демо RKNN
            cls_sum = torch.sum(cls, dim=1, keepdim=True)   # (B,1,H,W)
            cls_sum = torch.clamp(cls_sum, 0.0, 1.0)        # clip [0,1]
            outs += [reg, cls, cls_sum]
        return tuple(outs)

    def cleanup(self):
        for h in self._hooks:
            try: h.remove()
            except Exception: pass
        self._hooks = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640],
                        help='image size, the order is: height width')  # H, W
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--dynamic-batch', action='store_true', help='export dynamic batch onnx model')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--trt-version', type=int, default=8, help='tensorrt version')
    parser.add_argument('--ort', action='store_true', help='export onnx for onnxruntime')
    parser.add_argument('--with-preprocess', action='store_true', help='export bgr2rgb and normalize')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='conf threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # NEW:
    parser.add_argument('--rknn-opt', action='store_true', default=True,
                        help='export in RKNN-optimized style (9 outputs)')

    # you may force opset 13 for RKNN
    parser.add_argument('--opset', type=int, default=13)

    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()

    # Device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'

    # Load PyTorch model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)  # FP32
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
            layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)

    # Update model
    if args.half:
        img, model = img.half(), model.half()  # to FP16
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations (как в твоём оригинале)
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace

    dynamic_axes = None
    if args.dynamic_batch:
        args.batch_size = 'batch'
        dynamic_axes = {'images': {0: 'batch'},}

    # End2End (оригинальный путь) — оставляем как есть
    if args.end2end:
        from yolov6.models.end2end import End2End
        model = End2End(model, max_obj=args.topk_all, iou_thres=args.iou_thres, score_thres=args.conf_thres,
                        device=device, ort=args.ort, trt_version=args.trt_version, with_preprocess=args.with_preprocess)

    # -------- RKNN-optimized export wrapper --------
    wrapper = None
    out_names = None
    if (not args.end2end) and args.rknn_opt:
        head, reg_preds, cls_preds, nl, nc = find_head_lists(model)
        assert head is not None, "Не найден модуль головы с reg_preds/cls_preds"
        wrapper = OnnxExportWrapper(model, reg_preds, cls_preds, nl)
        # имена выходов в порядке, который ждёт zoo
        scales = ('P3', 'P4', 'P5')[:nl]
        out_names = []
        for s in scales:
            out_names += [f'{s}_boxes', f'{s}_scores', f'{s}_scores_sum']

    print("===================")
    print(model)
    print("===================")

    # dry run
    _ = (wrapper(img) if wrapper is not None else model(img))

    # ONNX export
    try:
        LOGGER.info('\nStarting to export ONNX...')
        export_file = args.weights.replace('.pt', '_rknn.onnx') if args.rknn_opt and not args.end2end \
                      else args.weights.replace('.pt', '.onnx')

        with BytesIO() as f:
            torch.onnx.export(
                wrapper if wrapper is not None else model,
                img,
                f,
                verbose=False,
                opset_version=args.opset,
                training=torch.onnx.TrainingMode.EVAL,
                do_constant_folding=True,
                input_names=['images'],
                output_names=out_names if (wrapper is not None) else (
                    ['num_dets', 'det_boxes', 'det_scores', 'det_classes'] if args.end2end else ['outputs']
                ),
                dynamic_axes=dynamic_axes
            )
            f.seek(0)
            onnx_model = onnx.load(f)
            onnx.checker.check_model(onnx_model)

        if args.simplify:
            try:
                import onnxsim
                LOGGER.info('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                LOGGER.info(f'Simplifier failure: {e}')
        onnx.save(onnx_model, export_file)
        LOGGER.info(f'ONNX export success, saved as {export_file}')
    except Exception as e:
        LOGGER.info(f'ONNX export failure: {e}')
    finally:
        if wrapper is not None:
            wrapper.cleanup()

    # Finish
    LOGGER.info('\nExport complete (%.2fs)' % (time.time() - t))
    if args.end2end and not args.ort:
        info = f'trtexec --onnx={export_file} --saveEngine={export_file.replace(".onnx",".engine")}'
        if args.dynamic_batch:
            LOGGER.info('Dynamic batch export should define min/opt/max batchsize\nWe set min/opt/max = 1/16/32 default!')
            wandh = 'x'.join(list(map(str, args.img_size)))
            info += (f' --minShapes=images:1x3x{wandh}'+
                     f' --optShapes=images:16x3x{wandh}'+
                     f' --maxShapes=images:32x3x{wandh}'+
                     f' --shapes=images:16x3x{wandh}')
        LOGGER.info('\nYou can export tensorrt engine use trtexec tools.\nCommand is:')
        LOGGER.info(info)
