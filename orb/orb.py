import cv2
import numpy as np
import time
from collections import deque

# ===== ПАРАМЕТРЫ =====
TARGET_H, TARGET_W = 480, 640     # целевой размер кадра (HxW)
VIDEO_SRC = 0                     # 0 — веб-камера. Либо путь к файлу, rtsp://..., и т.п.
USE_GSTREAMER = False             # True — открыть через GStreamer (HW-декодер RK3588)

# ===== ВСПОМОГАТЕЛЬНОЕ: letterbox без искажений =====
def letterbox(img, target_h, target_w, color=(0, 0, 0)):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Создаём полотно и вклеиваем по центру
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas, scale, (left, top), (new_w, new_h)

# ===== ОТКРЫТИЕ ВИДЕО =====
def open_capture():
    if USE_GSTREAMER:
        # Пример для H264/H265 файла/потока с аппаратным декодированием
        # Подставьте свой источник в location=...
        pipeline = (
            "filesrc location=video.mp4 ! "
            "qtdemux ! h264parse ! v4l2h264dec capture-io-mode=dmabuf ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(VIDEO_SRC)

cap = open_capture()
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видеоисточник")

# ===== ORB =====
# Параметры можно подстроить под вашу задачу (число особых точек, WTA_K и т.д.)
orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

# ===== FPS (скользящее среднее) =====
fps_hist = deque(maxlen=30)
prev_t = time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Подготовка кадра (letterbox)
    framed, scale, (offx, offy), (new_w, new_h) = letterbox(frame, TARGET_H, TARGET_W)

    # === Время начала обработки
    t0 = time.perf_counter()

    # Детектор ORB — по изображению после letterbox (можно и по оригиналу, но для визуализации логично по framed)
    gray = cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY)
    keypoints = orb.detect(gray, None)
    keypoints, descriptors = orb.compute(gray, keypoints)

    # Визуализация ключевых точек
    out = cv2.drawKeypoints(framed, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # === Подсчёт FPS обработки (время на один кадр)
    t1 = time.perf_counter()
    dt = t1 - t0
    inst_fps = 1.0 / dt if dt > 0 else 0.0
    fps_hist.append(inst_fps)
    smooth_fps = sum(fps_hist) / len(fps_hist)

    # Оверлеи
    cv2.putText(out, f"Frame proc FPS: {inst_fps:5.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(out, f"Avg(30) FPS:   {smooth_fps:5.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(out, f"Keypoints: {len(keypoints) if keypoints is not None else 0}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("ORB on RK3588 (letterbox)", out)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
