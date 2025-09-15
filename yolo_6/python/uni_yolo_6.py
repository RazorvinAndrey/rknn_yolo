# --- CUDA allocator config ДО импорта torch ---
import os
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.7"
)

import argparse
import time
from pathlib import Path
import math

import cv2
import numpy as np
import torch

# --- Патч torch.load для PyTorch >= 2.6 (YOLOv6 чекпойнты) ---
_ORIG_TORCH_LOAD = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # грузим «по-старому», без песочницы
    return _ORIG_TORCH_LOAD(*args, **kwargs)
torch.load = _patched_torch_load

if not torch.cuda.is_available():
    raise SystemExit("CUDA не обнаружена. Скрипт работает только на GPU.")

torch.backends.cudnn.benchmark = False  # экономим память
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
cv2.setNumThreads(0)

# --- YOLOv6 импорты (скрипт должен лежать в корне репозитория YOLOv6) ---
from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.events import load_yaml  # для чтения data.yaml с именами классов

# ----- letterbox: ЖЁСТКО до 640×640 (с сохранением пропорций) -----
def letterbox_exact(img_bgr, new_size=640, color=(114,114,114)):
    if isinstance(new_size, int):
        new_w = new_h = new_size
    else:
        new_w, new_h = new_size
    h, w = img_bgr.shape[:2]
    r = min(new_w / w, new_h / h)
    uw, uh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_bgr, (uw, uh), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_w - uw, new_h - uh
    left, right = dw // 2, dw - dw // 2
    top, bottom = dh // 2, dh - dh // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out  # BGR, ровно (640,640)

# ----- препроцесс: CPU tensor (channels_last + pinned) -----
def preprocess_bgr_to_cpu_tensor(img_bgr, img_size, half):
    lb_img = letterbox_exact(img_bgr, new_size=(img_size, img_size))   # BGR 640x640
    im = lb_img[:, :, ::-1].transpose(2, 0, 1).copy()                  # RGB CHW
    im = torch.from_numpy(im)
    im = im.half() if half else im.float()
    im /= 255.0
    im = im.unsqueeze(0)                                               # [1,3,640,640] CPU
    im = im.contiguous(memory_format=torch.channels_last).pin_memory()
    return im, lb_img

# ----- рисуем боксы + подписи -----
def draw_boxes(img_bgr, det, names=None):
    if det is None or len(det) == 0:
        return
    h, w = img_bgr.shape[:2]
    lw = max(round((h + w) / 2 * 0.003), 2)
    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), lw, cv2.LINE_AA)
        c = int(cls)
        label = f"{(names[c] if names and c < len(names) else f'cls{c}')} {conf:.2f}"
        tf = max(lw - 1, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lw/3, tf)
        outside = y1 - th - 3 >= 0
        y_txt = y1 - 2 if outside else y1 + th + 2
        p2 = (x1 + tw, y1 - th - 3) if outside else (x1 + tw, y1 + th + 3)
        cv2.rectangle(img_bgr, (x1, y1 - th - 3 if outside else y1), p2, (0,255,0), -1, cv2.LINE_AA)
        cv2.putText(img_bgr, label, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, lw/3, (0,0,0), tf, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser("YOLOv6 (CUDA) — 640x640 letterbox, метки классов, устойчиво к OOM")
    ap.add_argument("-w", "--weights", required=True, type=str, help="путь к .pt весам YOLOv6")
    ap.add_argument("-s", "--source",  required=True, type=str, help="путь к видеофайлу")
    ap.add_argument("--yaml", type=str, default="", help="путь к data.yaml (для имён классов); опционально")
    ap.add_argument("--conf", type=float, default=0.35, help="порог уверенности (по умолчанию 0.35)")
    ap.add_argument("--iou",  type=float, default=0.50, help="порог IOU для NMS (по умолчанию 0.50)")
    ap.add_argument("--max-det", type=int, default=100, help="максимум детекций на кадр")
    ap.add_argument("--no-half", action="store_true", help="отключить FP16")
    args = ap.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    source  = Path(args.source).expanduser().resolve()
    if not weights.exists():
        raise SystemExit(f"веса не найдены: {weights}")
    if not source.exists():
        raise SystemExit(f"видео не найдено: {source}")

    device = torch.device("cuda:0")
    half = not args.no_half

    # Классы (если дали yaml) — иначе будут 'cls{ID}'
    names = None
    if args.yaml:
        yaml_path = Path(args.yaml).expanduser().resolve()
        if yaml_path.exists():
            try:
                names = load_yaml(str(yaml_path)).get("names", None)
            except Exception:
                names = None

    # --- грузим МОДЕЛЬ на CPU, потом переносим на GPU (экономия VRAM при загрузке) ---
    model = DetectBackend(str(weights), device="cpu")
    stride = int(model.stride)
    model.model.eval()
    if half:
        try:
            model.model.half()
        except Exception:
            model.model.float()
            half = False
    model.model.to(device, memory_format=torch.channels_last)
    model.device = device  # чтобы внутренние вызовы не тянули CPU

    # --- заранее выделяем ОДИН входной тензор на GPU (1x3x640x640) ---
    imgsz = 640
    gpu_in = torch.empty(
        (1, 3, imgsz, imgsz),
        dtype=torch.float16 if half else torch.float32,
        device=device
    ).to(memory_format=torch.channels_last)

    # прогрев
    with torch.inference_mode():
        _ = model(gpu_in)

    # видео
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"не удалось открыть видео: {source}")

    win = f"YOLOv6 | {weights.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, imgsz, imgsz)   # окно ровно 640×640

    fps_ema, alpha = 0.0, 0.9
    frame_i = 0
    print("Запуск... ESC или 'q' — выход.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Препроцесс: чётко до 640×640 (BGR) + CPU-тензор
        cpu_im, lb_bgr = preprocess_bgr_to_cpu_tensor(frame, img_size=imgsz, half=half)
        assert lb_bgr.shape[0] == imgsz and lb_bgr.shape[1] == imgsz, "letterbox_exact дал не 640×640"

        t0 = time.perf_counter()
        with torch.inference_mode():
            gpu_in.copy_(cpu_im, non_blocking=True)   # без аллокаций
            pred = model(gpu_in)                      # сырой предикт

        # На CPU и NMS (как в офиц. пайплайне YOLOv6)
        if isinstance(pred, (list, tuple)):
            pred_cpu = pred[0].detach().float().cpu()
        else:
            pred_cpu = pred.detach().float().cpu()

        # ожидается shape: [bs, N, 5+nc]
        if pred_cpu.ndim == 2:
            pred_cpu = pred_cpu.unsqueeze(0)
        det_list = non_max_suppression(
            pred_cpu, args.conf, args.iou, classes=None, agnostic=False, max_det=args.max_det
        )
        det = det_list[0] if isinstance(det_list, (list, tuple)) else det_list

        # Рисуем боксы + подписи на letterboxed кадре 640×640
        draw_boxes(lb_bgr, det, names=names)

        dt = time.perf_counter() - t0
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        fps_ema = inst_fps if fps_ema == 0 else alpha * fps_ema + (1 - alpha) * inst_fps
        cv2.putText(lb_bgr, f"FPS: {fps_ema:.1f}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # гарантируем показ ровно 640×640 (на всякий случай, если оконный менеджер ужмёт)
        if lb_bgr.shape[0] != imgsz or lb_bgr.shape[1] != imgsz:
            lb_bgr = cv2.resize(lb_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(win, lb_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

        # подчистка
        del cpu_im, pred_cpu, det
        frame_i += 1
        if frame_i % 10 == 0:
            torch.cuda.empty_cache()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
