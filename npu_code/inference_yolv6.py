import cv2
import time
import numpy as np
from rknn.api import RKNN

# ----------------------------
# Настройки
# ----------------------------
RKNN_PATH = 'yolov6_my.rknn'
VIDEO_PATH = 'input.mp4'   # или 0 для веб-камеры
CLASS_NAMES = ['plane_drone']  # один класс
CONF_THR = 0.35
IOU_THR  = 0.50
SHOW_WIN = True            # False для максимального FPS (без отрисовки)

# boxes в формате LTRB относительно центра ячейки? — см. ниже MULTIPLY_STRIDE
# Если боксы получаются слишком маленькими/большими — переключи этот флаг.
MULTIPLY_STRIDE = True

# ----------------------------
# Вспомогательные функции
# ----------------------------
def letterbox_resize(image, new_shape=640, color=(114,114,114)):
    """Масштабируем так, чтобы бОльшая сторона стала new_shape, сохраняем пропорции,
    и падаем до квадратных 640x640 (как в yolov6). Возвращаем: img_out, ratio, (dw, dh)."""
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=img.dtype)
    # центрируем
    dh = (new_shape[0] - nh) // 2
    dw = (new_shape[1] - nw) // 2
    canvas[dh:dh+nh, dw:dw+nw] = img
    return canvas, r, (dw, dh)

def nms_numpy(boxes, scores, iou_thr=0.5):
    """Простая NMS (xyxy). boxes: Nx4, scores: N."""
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        # IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def decode_outputs(outputs, conf_thr=0.35, multiply_stride=True):
    """
    outputs: список из 9 numpy-массивов:
      [P3_boxes(1,4,80,80), P3_scores(1,C,80,80), P3_sum(1,1,80,80),
       P4_boxes(1,4,40,40), P4_scores(1,C,40,40), P4_sum(1,1,40,40),
       P5_boxes(1,4,20,20), P5_scores(1,C,20,20), P5_sum(1,1,20,20)]
    Возвращает: boxes (N,4) в координатах 640x640, scores (N,), class_ids (N,)
    """
    strides = [8, 16, 32]
    grids = [(80, 80), (40, 40), (20, 20)]
    all_boxes = []
    all_scores = []
    all_cls = []

    for lvl in range(3):
        b_idx = lvl*3
        boxes = outputs[b_idx]       # (1,4,H,W)
        scores = outputs[b_idx+1]    # (1,C,H,W) уже sigmoid
        # sum = outputs[b_idx+2]     # (1,1,H,W) — можно использовать как objectness, но при C=1 не нужно
        _, c, h, w = scores.shape
        s = strides[lvl]

        # (4,H,W) -> (H,W,4)
        boxes = np.transpose(boxes[0], (1,2,0))
        # (C,H,W) -> (H,W,C)
        scores = np.transpose(scores[0], (1,2,0))

        # grid центры
        # center_x = (col+0.5)*s, center_y = (row+0.5)*s
        cols = np.arange(w)
        rows = np.arange(h)
        cx = (cols + 0.5) * s
        cy = (rows + 0.5) * s
        cx_grid, cy_grid = np.meshgrid(cx, cy)

        # Предполагаем LTRB от центра (anchor-free). Если это не так — см. флаг ниже.
        L = boxes[..., 0]
        T = boxes[..., 1]
        R = boxes[..., 2]
        B = boxes[..., 3]

        if multiply_stride:
            # Если модель выдаёт расстояния в "ячейках"
            L *= s; T *= s; R *= s; B *= s

        x1 = cx_grid - L
        y1 = cy_grid - T
        x2 = cx_grid + R
        y2 = cy_grid + B

        # Для C=1 score = scores[...,0]; для C>1 — берём max по классу
        if c == 1:
            scr = scores[..., 0]
            cls_id = np.zeros_like(scr, dtype=np.int32)
        else:
            cls_id = np.argmax(scores, axis=-1)
            scr = scores[np.arange(h)[:,None], np.arange(w)[None,:], cls_id]

        # Фильтрация по порогу
        mask = scr >= conf_thr
        if not np.any(mask):
            continue

        xyxy = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=-1)
        all_boxes.append(xyxy)
        all_scores.append(scr[mask])
        all_cls.append(cls_id[mask])

    if not all_boxes:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    class_ids = np.concatenate(all_cls, axis=0).astype(np.int32)
    # clip в пределах 0..640
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, 640)
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, 640)
    return boxes, scores, class_ids

def scale_boxes_back(xyxy, ratio, pad, orig_w, orig_h):
    """Из 640x640 letterbox-координат возвращаемся в координаты исходного кадра."""
    dw, dh = pad
    # снимаем паддинг
    xyxy[:, [0,2]] -= dw
    xyxy[:, [1,3]] -= dh
    # делим на масштаб
    xyxy /= ratio
    # клипим
    xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, orig_w-1)
    xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, orig_h-1)
    return xyxy

# ----------------------------
# Основной цикл
# ----------------------------
def main():
    cv2.setNumThreads(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print('Cannot open video:', VIDEO_PATH)
        return

    rknn = RKNN()
    ret = rknn.load_rknn(RKNN_PATH)
    if ret != 0:
        print('Load RKNN failed')
        return

    # На устройстве RK3588:
    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print('Init runtime failed')
        return

    t0 = time.time()
    frames = 0
    smoothed_fps = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        orig_h, orig_w = frame.shape[:2]

        # letterbox -> 640x640, BGR uint8
        img_in, ratio, (dw, dh) = letterbox_resize(frame, 640)

        # инференс
        t_infer0 = time.time()
        outputs = rknn.inference(inputs=[img_in])
        t_infer1 = time.time()

        # outputs: список numpy, ожидаем 9
        if len(outputs) != 9:
            print('Unexpected outputs:', [o.shape for o in outputs])
            break

        # постпроцесс (в 640x640)
        boxes, scores, cls_ids = decode_outputs(outputs, CONF_THR, MULTIPLY_STRIDE)

        # NMS (можно по классам, но для C=1 достаточно один раз)
        keep = nms_numpy(boxes, scores, IOU_THR)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # в координаты исходного кадра
        boxes = scale_boxes_back(boxes.copy(), ratio, (dw, dh), orig_w, orig_h)

        # fps
        frames += 1
        dt = t_infer1 - t_infer0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        if smoothed_fps is None:
            smoothed_fps = inst_fps
        else:
            smoothed_fps = smoothed_fps * 0.9 + inst_fps * 0.1

        if SHOW_WIN:
            # рисуем
            for (x1, y1, x2, y2), sc, cid in zip(boxes.astype(int), scores, cls_ids):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                label = f'{CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else cid}:{sc:.2f}'
                cv2.putText(frame, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f'FPS: {smoothed_fps:.1f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.imshow('RKNN YOLOv6', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        else:
            # Максимальная скорость: не рисуем, просто считаем fps по времени инференса
            if frames % 50 == 0:
                print(f'FPS ~ {smoothed_fps:.1f}')

    total_time = time.time() - t0
    print(f'Processed {frames} frames in {total_time:.2f}s, avg FPS ~ {frames/total_time:.1f}')

    cap.release()
    if SHOW_WIN:
        cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    main()
