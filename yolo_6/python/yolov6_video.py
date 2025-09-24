
import os
import cv2
import sys
import time
import argparse
import numpy as np
from collections import deque
from rknn.api import RKNN

# add path (как в вашем коде)
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper

# ---------------- настройки ----------------
IMG_SIZE = (640, 640)          # (width, height)
PAD_COLOR = (0, 0, 0)          # как в вашем фото-скрипте
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
# Имена классов будут подхватываться динамически из --names; если нет — class_{id}
NAMES = None
# -------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description='YOLOv6 RKNN video with NPU_CORE_0_1_2 + boxes + FPS')
    ap.add_argument('--rknn', required=True, help='путь к .rknn модели')
    ap.add_argument('--video', required=True, help='путь к видео или индекс камеры (например, 0)')
    ap.add_argument('--target', default='rk3588', help='платформа (rk3588)')
    ap.add_argument('--device_id', default=None, help='ID устройства (если нужно)')
    ap.add_argument('--warmup', type=int, default=5, help='число прогревочных прогонов')
    ap.add_argument('--avg', type=int, default=60, help='окно усреднения FPS')
    ap.add_argument('--no_post', action='store_true',
                    help='выключить постпроцессинг (замер только препроцесс+инференс)')
    ap.add_argument('--no_display', action='store_true',
                    help='не показывать окно (можно оставить постпроцесс)')
    ap.add_argument('--names', type=str, default=None,
                    help='путь к txt с именами классов (по одному имени на строку)')
    return ap.parse_args()

# ----------- функции постпроцесса (CPU) -----------
def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1).astype(np.int32)  # <- гарантируем int
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos].astype(np.int32)                   # <- ещё раз явно
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]; y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]; h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        w1 = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h1 = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

# -------- DFL на NumPy (без torch) --------
def dfl_numpy(position):
    # position: (N, C, H, W), где C = 4 * mc
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y_max = y.max(axis=2, keepdims=True)
    e = np.exp(y - y_max)
    y = e / (e.sum(axis=2, keepdims=True) + 1e-12)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc).sum(axis=2)  # (N, 4, H, W)
    return y

# кэш сеток
_grid_cache = {}  # ключ: (grid_h, grid_w) -> (grid, stride)

def _get_grid_and_stride(grid_h, grid_w):
    key = (grid_h, grid_w)
    if key in _grid_cache:
        return _grid_cache[key]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    row = row.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    grid = np.concatenate((col, row), axis=1)  # (1,2,H,W)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w], dtype=np.float32).reshape(1,2,1,1)
    _grid_cache[key] = (grid, stride)
    return grid, stride

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    grid, stride = _get_grid_and_stride(grid_h, grid_w)
    if position.shape[1] == 4:
        pos = position
    else:
        pos = dfl_numpy(position)
    box_xy  = grid + 0.5 - pos[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + pos[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)  # (N,4,H,W)
    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch

    # игнорируем score_sum
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)   # NCHW -> NHWC
        return _in.reshape(-1, ch)

    boxes        = np.concatenate([sp_flatten(v) for v in boxes])           # (N, 4)
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])    # (N, C)
    scores       = np.concatenate([sp_flatten(v) for v in scores])          # (N, 1)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    if boxes is None or boxes.size == 0:
        return None, None, None

    nboxes, nclasses, nscores = [], [], []
    unique_cls = np.unique(classes)
    for cls_id in unique_cls:
        inds = np.where(classes == cls_id)[0]
        b = boxes[inds]; s = scores[inds]; cls_arr = classes[inds]
        if b.size == 0:
            continue
        keep = nms_boxes(b, s)
        if keep.size == 0:
            continue
        nboxes.append(b[keep]); nclasses.append(cls_arr[keep]); nscores.append(s[keep])

    if not nboxes:
        return None, None, None

    boxes   = np.concatenate(nboxes, axis=0)
    classes = np.concatenate(nclasses, axis=0).astype(np.int32)
    scores  = np.concatenate(nscores, axis=0)
    return boxes, classes, scores

def _class_name(cls_id: int) -> str:
    global NAMES
    if NAMES and 0 <= cls_id < len(NAMES):
        return NAMES[cls_id]
    return f'class_{int(cls_id)}'

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        label = _class_name(int(cl))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, f'{label} {score:.2f}',
                    (top, max(0, left - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# ---------------------------------------------------------

def main():
    global NAMES
    args = parse_args()

    # загрузим имена классов при необходимости
    if args.names is not None and os.path.isfile(args.names):
        with open(args.names, 'r', encoding='utf-8') as f:
            NAMES = [ln.strip() for ln in f if ln.strip() != '']
        # пустой список не нужен
        if len(NAMES) == 0:
            NAMES = None

    # RKNN init + все ядра NPU
    rknn = RKNN(verbose=False)
    print('--> Loading RKNN')
    ret = rknn.load_rknn(args.rknn)
    if ret != 0:
        print('Load RKNN failed'); sys.exit(ret)
    print('done')

    print('--> Init runtime (NPU_CORE_0_1_2)')
    ret = rknn.init_runtime(
        target=args.target,
        device_id=args.device_id,
        core_mask=RKNN.NPU_CORE_0_1_2
    )
    if ret != 0:
        print('Init runtime failed'); sys.exit(ret)
    print('done')

    # видео-источник
    try:
        cap_src = int(args.video)
    except ValueError:
        cap_src = args.video
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print('Open video failed'); rknn.release(); sys.exit(1)

    co_helper = COCO_test_helper(enable_letter_box=True)

    # прогрев
    print('--> Warmup')
    dummy = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)  # H,W,C (RGB uint8)
    for _ in range(args.warmup):
        _ = rknn.inference(inputs=[dummy])
    print('done')

    print('--> Start video inference (ESC/q to quit) | no_post={} | no_display={}'.format(args.no_post, args.no_display))
    times = deque(maxlen=args.avg)
    total = 0
    last_report = time.time()

    try:
        while True:
            ok, img_bgr = cap.read()
            if not ok:
                break

            t0 = time.time()

            # препроцесс (как в вашем фото-скрипте)
            img_ltr = co_helper.letter_box(im=img_bgr.copy(),
                                           new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
                                           pad_color=PAD_COLOR)
            img_rgb = cv2.cvtColor(img_ltr, cv2.COLOR_BGR2RGB)
            if img_rgb.dtype != np.uint8:
                img_rgb = img_rgb.astype(np.uint8)

            # инференс (NHWC uint8)
            outputs = rknn.inference(inputs=[img_rgb])

            frame_to_show = img_bgr
            if not args.no_post:
                # постпроцесс и (возможно) отрисовка
                boxes, classes, scores = post_process(outputs)
                if (boxes is not None) and (not args.no_display):
                    real_boxes = co_helper.get_real_box(boxes)
                    draw(frame_to_show, real_boxes, scores, classes)

            t1 = time.time()
            dt = t1 - t0
            times.append(dt)
            total += 1

            # FPS
            if t1 - last_report >= 1.0:
                inst_fps = 1.0 / dt if dt > 0 else 0.0
                avg_dt = sum(times) / len(times) if times else 0.0
                avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
                print(f'Frames: {total} | Instant FPS: {inst_fps:6.2f} | Avg FPS({len(times)}): {avg_fps:6.2f}')
                last_report = t1

            # показ окна (если не отключён)
            if not args.no_display:
                cv2.imshow('YOLOv6 RKNN (NPU 0_1_2)', frame_to_show)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        rknn.release()
        print('--> Done')

if __name__ == '__main__':
    main()
