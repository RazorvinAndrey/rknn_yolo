#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

# ----------------- helpers -----------------

class HeatmapBuilder:
    def __init__(self, inh, inw, r=8):
        self.inh = inh
        self.inw = inw
        self.r = r
        # precompute index maps once (for speed)
        Hc, Wc = inh // r, inw // r
        ys = np.arange(inh); xs = np.arange(inw)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        self.yy = yy
        self.xx = xx
        self.yc = yy // r
        self.xc = xx // r
        self.ci_yx = (yy % r) * r + (xx % r)   # "YX" (классика)
        self.ci_xy = (xx % r) * r + (yy % r)   # альтернативный порядок
        # маски на случай граничных несовпадений размеров (бывает у некоторых экспортеров)
        self.Hc = Hc; self.Wc = Wc

    def _heat_from_prob64_nchw(self, prob64, order='yx'):
        # prob64: [1,64,Hc,Wc] или [64,Hc,Wc]
        p = prob64[0] if prob64.ndim == 4 else prob64  # [64,Hc,Wc]
        ci = self.ci_yx if order == 'yx' else self.ci_xy
        heat = p[ci, self.yc, self.xc]  # [H,W]
        return np.clip(heat, 0.0, 1.0).astype(np.float32)

    def _heat_from_prob64_nhwc(self, prob64, order='yx'):
        # prob64: [1,Hc,Wc,64] или [Hc,Wc,64]
        p = prob64[0] if prob64.ndim == 4 else prob64  # [Hc,Wc,64]
        ci = self.ci_yx if order == 'yx' else self.ci_xy
        heat = p[self.yc, self.xc, ci]  # [H,W]
        return np.clip(heat, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def softmax_axis(x, axis):
        x = x.astype(np.float32)
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        return e / np.sum(e, axis=axis, keepdims=True)

    def build_heat(self, semi, layout='auto', order='auto'):
        a = np.asarray(semi)
        if a.ndim == 3:
            a = a[None, ...]
        if a.ndim != 4:
            raise RuntimeError(f'semi has unexpected shape {a.shape}')

        # detect channel axis (==65)
        axes65 = [i for i, s in enumerate(a.shape) if s == 65]
        if not axes65:
            raise RuntimeError(f'no axis==65 in shape {a.shape}')
        c_axis = axes65[0]

        # softmax over 65
        prob = self.softmax_axis(a, axis=c_axis)

        # drop dustbin
        slicer = [slice(None)] * 4
        slicer[c_axis] = slice(0, 64)
        prob64 = prob[tuple(slicer)]

        # choose layout
        layouts = ['nchw', 'nhwc'] if layout == 'auto' else [layout]
        orders = ['yx', 'xy'] if order == 'auto' else [order]

        # try candidates, score each by "non-gridness": больше уникальных пиков → лучше
        best = None
        best_score = -1
        best_heat = None
        for L in layouts:
            for O in orders:
                try:
                    if L == 'nchw':    # expect prob64 as [N,64,Hc,Wc] or [64,Hc,Wc]
                        heat = self._heat_from_prob64_nchw(prob64, order=O)
                    else:              # 'nhwc': [N,Hc,Wc,64] или [Hc,Wc,64]
                        # если текущая раскладка не NHWC — нужно транспонировать
                        if c_axis == 1:  # мы получили NCHW; преобразуем в NHWC
                            if prob64.ndim == 4:
                                # [N,64,Hc,Wc] -> [N,Hc,Wc,64]
                                prob64_nhwc = np.transpose(prob64, (0, 2, 3, 1))
                            else:
                                prob64_nhwc = np.transpose(prob64, (1, 2, 0))
                        else:
                            prob64_nhwc = prob64
                        heat = self._heat_from_prob64_nhwc(prob64_nhwc, order=O)

                    # оценка: берём локальные максимумы и считаем их «плотность»/энтропию
                    ksize = 9
                    dil = cv2.dilate(heat, np.ones((ksize, ksize), np.uint8))
                    lm = (heat == dil) & (heat > 0.01)
                    score = int(lm.sum())
                except Exception:
                    continue

                if score > best_score:
                    best_score = score
                    best = (L, O)
                    best_heat = heat

        if best is None:
            raise RuntimeError('failed to resolve layout/order automatically')

        return best_heat, best  # heatmap [H,W], (layout, order)

def nms_and_topk(heatmap, nms_radius=4, conf_thresh=0.015, topk=1000,
                 _cache={}):
    # cache kernel by radius
    key = ('nmskernel', nms_radius)
    if key not in _cache:
        _cache[key] = np.ones((2 * nms_radius + 1, 2 * nms_radius + 1), np.uint8)
    kernel = _cache[key]
    dilated = cv2.dilate(heatmap, kernel)
    local_max = (heatmap == dilated) & (heatmap > conf_thresh)
    ys, xs = np.where(local_max)
    scores = heatmap[ys, xs]
    if scores.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)
    if topk and scores.size > topk:
        idx = np.argpartition(-scores, topk)[:topk]
        ys, xs, scores = ys[idx], xs[idx], scores[idx]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    return pts, scores

def draw_keypoints(img_bgr, pts, color=(0, 255, 0)):
    for x, y in pts.astype(int):
        cv2.circle(img_bgr, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    return img_bgr

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rknn', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--inh', type=int, required=True)
    ap.add_argument('--inw', type=int, required=True)
    ap.add_argument('--thr', type=float, default=0.015)
    ap.add_argument('--nms', type=int, default=4)
    ap.add_argument('--topk', type=int, default=1000)
    ap.add_argument('--out', type=str, default='')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--cores', type=str, default='all', choices=['0', '1', '2', 'all'])
    # форс-настройки (по умолчанию автоопределение)
    ap.add_argument('--force_layout', choices=['nchw', 'nhwc'], default=None)
    ap.add_argument('--force_order', choices=['yx', 'xy'], default=None)
    args = ap.parse_args()

    rk = RKNNLite()
    assert rk.load_rknn(args.rknn) == 0, 'Failed to load RKNN'
    core_mask = RKNNLite.NPU_CORE_0_1_2 if args.cores == 'all' else \
                (RKNNLite.NPU_CORE_0 if args.cores == '0' else
                 RKNNLite.NPU_CORE_1 if args.cores == '1' else RKNNLite.NPU_CORE_2)
    assert rk.init_runtime(core_mask=core_mask) == 0, 'Failed to init runtime'

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {args.video}')

    # немного ускоряем вывод
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, src_fps, (src_w, src_h))

    print(f'[INFO] Video: {src_w}x{src_h}@{src_fps:.2f} | Model input: {args.inw}x{args.inh}')

    hb = HeatmapBuilder(args.inh, args.inw, r=8)

    # авто-калибровка раскладки на первом кадре
    layout_fixed = args.force_layout
    order_fixed  = args.force_order
    resolved = None  # (layout, order) chosen

    t0 = time.time()
    frame_count = 0
    fps_smooth = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # подача в NPU: uint8, если при сборке стоял std=255
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            inp_resized = cv2.resize(gray, (args.inw, args.inh), interpolation=cv2.INTER_AREA)
            inp = inp_resized[None, None, :, :].astype(np.uint8)

            t_start = time.time()
            outputs = rk.inference(inputs=[inp])

            # ищем тензор с размерностью 65
            semi = None
            for out in outputs:
                if 65 in np.asarray(out).shape:
                    semi = out
                    break
            if semi is None:
                raise RuntimeError(f'No semi tensor among outputs: {[np.asarray(o).shape for o in outputs]}')

            semi = np.asarray(semi)  # может быть uint8/int8/float
            if semi.dtype != np.float32 and semi.dtype != np.float16:
                # если RKNN вернул квантизованные логиты — просто приводим к float32,
                # softmax нивелирует масштаб/сдвиг (достаточно для поиска максимумов)
                semi = semi.astype(np.float32)

            # строим теплокарту
            if resolved is None and (layout_fixed is None or order_fixed is None):
                heat_auto, (L, O) = hb.build_heat(semi,
                                                  layout=layout_fixed or 'auto',
                                                  order=order_fixed or 'auto')
                resolved = (L, O)
                heat = heat_auto
                print(f'[INFO] Resolved layout/order: {resolved}')
            else:
                # уже знаем раскладку — быстрое построение
                L, O = layout_fixed, order_fixed
                # повторяем часть build_heat, но без перебора
                a = semi
                if a.ndim == 3: a = a[None, ...]
                axes65 = [i for i, s in enumerate(a.shape) if s == 65]
                c_axis = axes65[0]
                prob = hb.softmax_axis(a, axis=c_axis)
                slicer = [slice(None)] * 4
                slicer[c_axis] = slice(0, 64)
                prob64 = prob[tuple(slicer)]
                if L == 'nchw':
                    heat = hb._heat_from_prob64_nchw(prob64, order=O)
                else:
                    if c_axis == 1:
                        prob64 = np.transpose(prob64, (0, 2, 3, 1)) if prob64.ndim == 4 else np.transpose(prob64, (1,2,0))
                    heat = hb._heat_from_prob64_nhwc(prob64, order=O)

            pts_model, scores = nms_and_topk(heat, nms_radius=args.nms, conf_thresh=args.thr, topk=args.topk)

            # к исходному размеру
            if pts_model.size:
                pts_src = pts_model.copy()
                pts_src[:, 0] *= (src_w / float(args.inw))
                pts_src[:, 1] *= (src_h / float(args.inh))
            else:
                pts_src = pts_model

            dt = time.time() - t_start
            fps_inst = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = fps_inst if fps_smooth is None else 0.9 * fps_smooth + 0.1 * fps_inst

            draw_keypoints(frame_bgr, pts_src, (0, 255, 0))
            cv2.putText(frame_bgr, f'FPS: {fps_smooth:.1f}  KPs: {len(pts_src)}',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 230, 50), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame_bgr)
            if args.show:
                cv2.imshow('SuperPoint RKNN', frame_bgr)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break

            frame_count += 1

    finally:
        cap.release()
        if writer is not None: writer.release()
        cv2.destroyAllWindows()
        rk.release()

    total = time.time() - t0
    print(f'[INFO] Done. Frames: {frame_count}, Avg FPS: {frame_count/total:.2f}')
    if resolved:
        print(f'[INFO] Final layout/order used: {resolved}')

if __name__ == '__main__':
    main()
