#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified SuperPoint video tester: PyTorch (.pth) / ONNX / RKNN
- draws keypoints on frames
- prints FPS
- SAME post-processing for all backends (canonical SuperPoint)

Usage examples:

# 1) PyTorch (на ПК)
python sp_video_test.py --backend torch --pth superpoint_v1.pth \
    --video input.mp4 --inh 480 --inw 640 --show

# 2) ONNX (на ПК)
python sp_video_test.py --backend onnx --onnx superpoint_480x640.onnx \
    --video input.mp4 --inh 480 --inw 640 --show

# 3) RKNN (на RK3588)
python sp_video_test.py --backend rknn --rknn superpoint_480x640.rknn \
    --video input.mp4 --inh 480 --inw 640 --rknn_input uint8 --show

Notes:
- --inh/--inw должны соответствовать размеру, с которым экспортировалась модель.
- Ожидается NCHW: semi [1,65,Hc,Wc], desc [1,256,Hc,Wc]. Если порядок выходов иной, укажи --semi_idx/--desc_idx.
"""

import os
import time
import argparse
import numpy as np
import cv2

# ---------- Common post-processing (canonical SuperPoint) ----------

def softmax_np(x, axis=1):
    x = x.astype(np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)

def semi_to_heatmap_nchw(semi_nchw, r=8):
    """
    semi_nchw: np.ndarray [1,65,Hc,Wc] (logits).
    Returns heatmap [H, W] float32 in [0,1].
    Canonical mapping from original SuperPoint:
      prob = softmax(semi, dim=1)[:, :64]
      prob -> [N, 8, 8, Hc, Wc] -> permute to [N, Hc, 8, Wc, 8] -> reshape [N, 1, H, W]
    """
    assert semi_nchw.ndim == 4 and semi_nchw.shape[1] == 65, f"Expected [1,65,Hc,Wc], got {semi_nchw.shape}"
    prob = softmax_np(semi_nchw, axis=1)[:, :64, :, :]   # [1,64,Hc,Wc]
    b, c, hc, wc = prob.shape
    # [1,64,Hc,Wc] -> [1,8,8,Hc,Wc]
    prob = prob.reshape(b, 8, 8, hc, wc)
    # [1,8,8,Hc,Wc] -> [1,Hc,8,Wc,8]
    prob = np.transpose(prob, (0, 3, 1, 4, 2))
    # -> [1, Hc*8, Wc*8, 1] -> [1,1,H,W] -> [H,W]
    heat = prob.reshape(b, hc * r, wc * r, 1)
    heat = np.transpose(heat, (0, 3, 1, 2))[0, 0]
    return heat.astype(np.float32)

def nms_and_topk(heatmap, nms_radius=4, conf_thresh=0.015, topk=1000):
    k = 2 * nms_radius + 1
    dil = cv2.dilate(heatmap, np.ones((k, k), np.uint8))
    local_max = (heatmap == dil) & (heatmap > conf_thresh)
    ys, xs = np.where(local_max)
    scores = heatmap[ys, xs]
    if scores.size == 0:
        return np.empty((0,2), np.float32), scores
    if topk and scores.size > topk:
        idx = np.argpartition(-scores, topk)[:topk]
        ys, xs, scores = ys[idx], xs[idx], scores[idx]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    return pts, scores

def draw_keypoints(img_bgr, pts, color=(0,255,0)):
    for x, y in pts.astype(int):
        cv2.circle(img_bgr, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    return img_bgr

# ---------- NEW: Letterbox (сохранение пропорций + чёрные поля) ----------

def letterbox_bgr(img_bgr, target_w, target_h):
    """
    Возвращает:
      canvas_bgr (target_h x target_w x 3) — исходный кадр вписан по длинной стороне, остальное — чёрное,
      gray_canvas (target_h x target_w) — ч/б версия canvas для подачи в модель.
    """
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=img_bgr.dtype)
    pad_left = (target_w - new_w) // 2
    pad_top  = (target_h - new_h) // 2
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return canvas, gray_canvas

# ---------- Backends ----------

class TorchBackend:
    def __init__(self, pth_path, device='cpu'):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.device = device

        # Minimal canonical SuperPoint architecture (no BN)
        class SuperPointNet(nn.Module):
            def __init__(self):
                super().__init__()
                relu = nn.ReLU(inplace=True)
                c1 = [nn.Conv2d(1, 64, 3, padding=1), relu,
                      nn.Conv2d(64, 64, 3, padding=1), relu,
                      nn.MaxPool2d(2,2)]  # /2
                c2 = [nn.Conv2d(64, 64, 3, padding=1), relu,
                      nn.Conv2d(64, 64, 3, padding=1), relu,
                      nn.MaxPool2d(2,2)]  # /4
                c3 = [nn.Conv2d(64, 128, 3, padding=1), relu,
                      nn.Conv2d(128,128,3, padding=1), relu,
                      nn.MaxPool2d(2,2)]  # /8
                c4 = [nn.Conv2d(128,128,3, padding=1), relu,
                      nn.Conv2d(128,128,3, padding=1), relu]
                self.encoder = nn.Sequential(*c1, *c2, *c3, *c4)
                # heads
                self.det_head = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1), relu,
                    nn.Conv2d(256, 65, 1)
                )
                self.desc_head = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1), relu,
                    nn.Conv2d(256, 256, 1)
                )

            def forward(self, x):
                fe = self.encoder(x)             # /8
                semi = self.det_head(fe)         # [N,65,Hc,Wc]
                desc = self.desc_head(fe)        # [N,256,Hc,Wc]
                # L2 normalize descriptors per location (optional)
                dn = self.torch.norm(desc, p=2, dim=1, keepdim=True) + 1e-8
                desc = desc / dn
                return semi, desc

        self.model = SuperPointNet().to(self.device).eval()

        # load weights (state_dict or scripted)
        sd = None
        try:
            ckpt = self.torch.load(pth_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            elif isinstance(ckpt, dict):
                sd = ckpt
        except Exception:
            ckpt = None

        if sd is not None:
            # allow non-strict to tolerate naming differences
            self.model.load_state_dict(sd, strict=False)
        else:
            # try TorchScript
            try:
                self.model = self.torch.jit.load(pth_path, map_location=self.device).eval()
            except Exception as e:
                raise RuntimeError(f"Cannot load {pth_path} as state_dict or TorchScript: {e}")

    def infer(self, img_float01_nchw):
        # img: np.float32 [1,1,H,W]
        x = self.torch.from_numpy(img_float01_nchw).to(self.device)
        with self.torch.no_grad():
            semi, desc = self.model(x)
        semi = semi.detach().cpu().numpy()
        desc = desc.detach().cpu().numpy()
        return semi, desc


class OnnxBackend:
    def __init__(self, onnx_path, semi_idx=0, desc_idx=1, force_nchw=True):
        import onnxruntime as ort
        self.ort = ort
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.semi_idx = semi_idx
        self.desc_idx = desc_idx
        self.force_nchw = force_nchw

    def infer(self, img_float01_nchw):
        ort_inputs = {self.input_name: img_float01_nchw.astype(np.float32)}
        outs = self.sess.run(None, ort_inputs)
        semi = outs[self.semi_idx]
        desc = outs[self.desc_idx]
        if self.force_nchw:
            assert semi.shape[1] == 65, f"Expected NCHW semi with C=65, got {semi.shape}"
            assert desc.shape[1] == 256, f"Expected NCHW desc with C=256, got {desc.shape}"
        return semi.astype(np.float32), desc.astype(np.float32)


class RKNNBackend:
    def __init__(self, rknn_path, semi_idx=0, desc_idx=1, input_type='uint8', cores='all'):
        from rknnlite.api import RKNNLite
        self.RKNNLite = RKNNLite
        self.rk = RKNNLite()
        ret = self.rk.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN")
        mask = (RKNNLite.NPU_CORE_0_1_2 if cores=='all' else
                RKNNLite.NPU_CORE_0 if cores=='0' else
                RKNNLite.NPU_CORE_1 if cores=='1' else
                RKNNLite.NPU_CORE_2)
        ret = self.rk.init_runtime(core_mask=mask)
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime")
        self.semi_idx = semi_idx
        self.desc_idx = desc_idx
        assert input_type in ('uint8','float32')
        self.input_type = input_type

    def infer(self, img_float01_nchw):
        if self.input_type == 'uint8':
            # предполагаем, что в rknn.config стояло std_values=[[255]] и mean=[[0]]
            inp = (img_float01_nchw * 255.0 + 0.5).astype(np.uint8)
        else:
            inp = img_float01_nchw.astype(np.float32)

        outs = self.rk.inference(inputs=[inp])
        semi = np.asarray(outs[self.semi_idx]).astype(np.float32)
        desc = np.asarray(outs[self.desc_idx]).astype(np.float32)
        # Ожидаем NCHW
        assert 65 in semi.shape and 256 in desc.shape, f"Unexpected RKNN output shapes: semi {semi.shape}, desc {desc.shape}"
        if semi.ndim == 4 and semi.shape[1] == 65:
            return semi, desc
        # если вдруг NHWC — явно скажем, без «магии»
        raise RuntimeError(f"RKNN outputs not NCHW (got {semi.shape}). Re-export with NCHW or add a layout transpose in conversion.")


# ---------- Main runner ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['torch','onnx','rknn'])
    ap.add_argument('--video', required=True)
    ap.add_argument('--inh', type=int, required=True)
    ap.add_argument('--inw', type=int, required=True)
    ap.add_argument('--thr', type=float, default=0.015)
    ap.add_argument('--nms', type=int, default=4)
    ap.add_argument('--topk', type=int, default=1000)
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--show', action='store_true')

    # Torch
    ap.add_argument('--pth', type=str, help='.pth (state_dict) или TorchScript .pt')
    ap.add_argument('--torch_device', default='cpu')

    # ONNX
    ap.add_argument('--onnx', type=str, help='path to .onnx')
    ap.add_argument('--onnx_semi_idx', type=int, default=0)
    ap.add_argument('--onnx_desc_idx', type=int, default=1)

    # RKNN
    ap.add_argument('--rknn', type=str, help='path to .rknn')
    ap.add_argument('--rknn_semi_idx', type=int, default=0)
    ap.add_argument('--rknn_desc_idx', type=int, default=1)
    ap.add_argument('--rknn_input', choices=['uint8','float32'], default='uint8',
                    help='uint8 (если std=255 в rknn.config) или float32 [0..1]')
    ap.add_argument('--rknn_cores', choices=['0','1','2','all'], default='all')

    args = ap.parse_args()

    if args.backend == 'torch':
        assert args.pth and os.path.exists(args.pth), "--pth required"
        backend = TorchBackend(args.pth, device=args.torch_device)
    elif args.backend == 'onnx':
        assert args.onnx and os.path.exists(args.onnx), "--onnx required"
        backend = OnnxBackend(args.onnx, semi_idx=args.onnx_semi_idx, desc_idx=args.onnx_desc_idx)
    else:
        assert args.rknn and os.path.exists(args.rknn), "--rknn required"
        backend = RKNNBackend(args.rknn, semi_idx=args.rknn_semi_idx, desc_idx=args.rknn_desc_idx,
                              input_type=args.rknn_input, cores=args.rknn_cores)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # --- CHANGED: пишем выход в размере модели (inw x inh) ---
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, src_fps, (args.inw, args.inh))

    print(f"[INFO] Output video size: {args.inw}x{args.inh} (letterbox). Backend: {args.backend}")

    t0 = time.time()
    frame_count = 0
    fps_smooth = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # --- CHANGED: Letterbox к размеру модели с чёрными полями ---
            canvas_bgr, gray_canvas = letterbox_bgr(frame_bgr, args.inw, args.inh)

            # --- Preprocess для модели: float32 [0,1] NCHW (уже нужного размера) ---
            img_nchw = gray_canvas[None, None, :, :].astype(np.float32) / 255.0

            t_start = time.time()
            semi, desc = backend.infer(img_nchw)  # expect NCHW shapes

            # --- Postprocess (canonical) ---
            heat = semi_to_heatmap_nchw(semi, r=8)
            pts_model, scores = nms_and_topk(heat, nms_radius=args.nms, conf_thresh=args.thr, topk=args.topk)

            # Рисуем точки прямо на уменьшенном кадре (canvas_bgr) — размеры совпадают с моделью
            frame_out = canvas_bgr.copy()
            draw_keypoints(frame_out, pts_model, (0,255,0))

            dt = time.time() - t_start
            fps_inst = 1.0/dt if dt>0 else 0.0
            fps_smooth = fps_inst if fps_smooth is None else 0.9*fps_smooth + 0.1*fps_inst

            cv2.putText(frame_out, f'FPS: {fps_smooth:.1f}  KPs: {len(pts_model)}',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,230,50), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame_out)
            if args.show:
                cv2.imshow('SuperPoint (letterboxed)', frame_out)
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q')):
                    break

            frame_count += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if args.backend == 'rknn':
            backend.rk.release()

    total = time.time() - t0
    print(f"[INFO] Done. Frames: {frame_count}, Avg FPS: {frame_count/total:.2f}")

if __name__ == '__main__':
    main()
