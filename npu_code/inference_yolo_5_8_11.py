import argparse
import os
import sys
import time
from pathlib import Path
import math
import cv2
import torch
import numpy as np

# --------- Глобальные настройки производительности ----------
torch.backends.cudnn.benchmark = True
try:
    # В новых torch можно повысить точность/скорость матмулов
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

cv2.setNumThreads(0)

# ---------------- Базовый интерфейс детектора ----------------
class BaseDetector:
    def __init__(self, weights_path: str, device: str = "cuda:0", half: bool = True):
        self.weights_path = weights_path
        self.device = device
        self.half = half

    def warmup(self):
        """Опциональный прогрев модели — переопределяется в наследниках."""
        pass

    def infer_and_draw(self, frame_bgr_640):
        """Должен вернуть кадр с наложенными боксами (BGR uint8, 640x640)."""
        raise NotImplementedError


# ---------------- Ultralytics YOLO (v8/v11/v12, а также совместимые v5) ----------------
class UltralyticsDetector(BaseDetector):
    def __init__(self, weights_path: str, device: str = "cuda:0", half: bool = True):
        super().__init__(weights_path, device, half)
        from ultralytics import YOLO  # импорт здесь, чтобы не требовать пакет без надобности

        self.model = YOLO(self.weights_path)
        # Принудительно выставляем девайс
        self.model.to(self.device)
        self.use_half = False
        # half для ultralytics включается через .predict(..., half=True)
        # далеко не все веса/опции корректно работают в half, дадим опцию на вызове

    def warmup(self):
        # Прогрев: пустой прогон 1×3×640×640
        import numpy as np
        dummy = (np.zeros((640, 640, 3), dtype=np.uint8))
        _ = self.model.predict(
            dummy,
            imgsz=640,
            device=self.device,
            half=self.half,
            verbose=False,
        )

    def infer_and_draw(self, frame_bgr_640):
        # Возвращаем кадр с аннотациями от самой модели
        results = self.model.predict(
            frame_bgr_640,
            imgsz=640,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        # results[0].plot() -> annotated image (BGR)
        return results[0].plot()


# ---------------- TorchHub (YOLOv5/v7/v9/возможно v6/v10 если есть hubconf) ----------------
class TorchHubDetector(BaseDetector):
    def __init__(self, repo: str, weights_path: str, device: str = "cuda:0", half: bool = True):
        super().__init__(weights_path, device, half)
        self.repo = repo
        # Для torch.hub нужна установленная git
        try:
            self.model = torch.hub.load(self.repo, 'custom', path=self.weights_path, source='github')
        except Exception as e:
            raise RuntimeError(
                f"Не удалось загрузить модель из Torch Hub ({self.repo}). "
                f"Проверьте наличие hubconf.py в репозитории или установите модельным "
                f"репозиторием напрямую. Исходная ошибка: {e}"
            )
        if self.device.startswith("cuda"):
            self.model.to(self.device)
            if self.half:
                try:
                    self.model.half()
                except Exception:
                    # Если half недоступен — продолжаем в FP32
                    pass
        else:
            raise RuntimeError("Скрипт поддерживает только CUDA. Включите GPU.")

        # Доп. настройки (если поддерживаются моделью yolov5-подобного API)
        try:
            self.model.conf = 0.25  # порог по умолчанию
            self.model.iou = 0.45
        except Exception:
            pass

    def warmup(self):
        # У yolov5-подобного API warmup необязателен, сделаем один холостой прогон
        import numpy as np
        dummy = (np.zeros((640, 640, 3), dtype=np.uint8))
        try:
            _ = self.model(dummy, size=640)
        except Exception:
            # Некоторые реализации требуют другой вызов — пропустим
            pass

    def infer_and_draw(self, frame_bgr_640):
        # Для yolov5-подобного API: results.render() вернёт аннотированные кадры
        results = self.model(frame_bgr_640, size=640)
        # results.render() может вернуть None и записать в results.imgs — учитываем оба варианта
        try:
            ims = results.render()  # list of images
            if isinstance(ims, list) and len(ims):
                return ims[0]
        except Exception:
            pass
        # Некоторые реализации возвращают уже отрисованный кадр в results.ims
        try:
            if hasattr(results, "imgs") and len(results.imgs):
                return results.imgs[0]
        except Exception:
            pass
        # Если API отличается, покажем ошибку
        raise RuntimeError(
            "TorchHubDetector: неизвестный формат результата. "
            "Вероятно, выбранный репозиторий имеет другой API отрисовки."
        )
def letterbox(img_bgr, new_size=640, color=(114, 114, 114)):
    """
    Масштабирует с сохранением пропорций и дополняет паддингом до new_size×new_size.
    Возвращает: img_out, (r_w, r_h), (dw, dh)
    """
    h, w = img_bgr.shape[:2]
    if isinstance(new_size, int):
        new_w = new_h = new_size
    else:
        new_w, new_h = new_size

    # коэффициент масштабирования (по меньшей стороне)
    r = min(new_w / w, new_h / h)
    # размер без паддинга
    uw, uh = int(round(w * r)), int(round(h * r))

    # собственно ресайз
    resized = cv2.resize(img_bgr, (uw, uh), interpolation=cv2.INTER_LINEAR)

    # паддинги по сторонам, чтобы получить ровно new_size×new_size
    dw = new_w - uw
    dh = new_h - uh
    left   = int(math.floor(dw / 2))
    right  = int(math.ceil (dw / 2))
    top    = int(math.floor(dh / 2))
    bottom = int(math.ceil (dh / 2))

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, (r, r), (left, top)


# ---------------- Фабрика по версиям YOLO ----------------
def make_detector(yolo_version: int, weights: str, device: str, half: bool):
    """
    Возвращает подходящий детектор для указанной версии.
    v5:         TorchHub (ultralytics/yolov5) ИЛИ Ultralytics (если веса совместимы)
    v8/11/12:   Ultralytics
    v7/v9:      TorchHub (если репозитории поддерживают hubconf)
    v6/v10:     Попытка TorchHub, если нет — бросаем понятную ошибку.
    """
    v = int(yolo_version)
    weights = str(weights)

    if v in (8, 11, 12):
        return UltralyticsDetector(weights, device=device, half=half)

    if v == 5:
        # Сначала попробуем Ultralytics (иногда удобно), затем TorchHub
        try:
            return UltralyticsDetector(weights, device=device, half=half)
        except Exception:
            return TorchHubDetector("ultralytics/yolov5", weights, device=device, half=half)

    if v == 7:
        return TorchHubDetector("WongKinYiu/yolov7", weights, device=device, half=half)

    if v == 9:
        return TorchHubDetector("WongKinYiu/yolov9", weights, device=device, half=half)

    if v == 6:
        # Некоторые сборки YOLOv6 не имеют hubconf. Пробуем, иначе подскажем.
        try:
            return TorchHubDetector("meituan/YOLOv6", weights, device=device, half=half)
        except Exception as e:
            raise RuntimeError(
                "YOLOv6: Похоже, ваш формат/репозиторий не поддерживает TorchHub.\n"
                "Решения: "
                "1) запустите инференс из офиц. репозитория YOLOv6; "
                "2) экспортируйте веса в ONNX/TensorRT; "
                "3) конвертируйте в совместимый формат Ultralytics."
            ) from e

    if v == 10:
        # Аналогично — многие сборки не имеют hubconf.
        try:
            return TorchHubDetector("THU-MIG/yolov10", weights, device=device, half=half)
        except Exception as e:
            raise RuntimeError(
                "YOLOv10: TorchHub API, вероятно, отсутствует в вашей сборке.\n"
                "Решения: запустите из офиц. репозитория YOLOv10 или экспортируйте/сконвертируйте веса."
            ) from e

    raise ValueError(f"Неизвестная версия YOLO: v{v}")


# ----------------- Основной цикл видео-инференса -----------------
def run(video_path: str, detector: BaseDetector, window_name: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    # Быстрый прогрев
    detector.warmup()

    fps_ema = 0.0
    alpha = 0.9  # EMA для сглаживания FPS
    fpss = []
    print("Запуск... Нажмите ESC или 'q' для выхода.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Масштабируем вход в 640×640 (строгое требование)
        frame_640, _, _ = letterbox(frame, 640)

        t0 = time.perf_counter()
        annotated = detector.infer_and_draw(frame_640)
        dt = time.perf_counter() - t0
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        #fps_ema = (alpha * fps_ema + (1 - alpha) * inst_fps) if fps_ema > 0 else inst_fps
        if inst_fps > 0:
            fpss.append(inst_fps)
        # Рисуем FPS
        txt = f"FPS: {inst_fps:.1f}"
        cv2.putText(annotated, txt, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):  # ESC / q
            break
    arr = np.array(fpss)
    cap.release()
    cv2.destroyAllWindows()
    print(f'среднее значение FPS = {arr.mean()}')


def main():
    parser = argparse.ArgumentParser(
        description="Быстрый запуск YOLO (v5–v12) на CUDA: выбор весов, версии и видео. Без записи, только показ."
    )
    parser.add_argument("--version", "-v", type=int, help="Версия YOLO: 5/6/7/8/9/10/11/12")
    parser.add_argument("--weights", "-w", type=str, help="Путь к .pt весам")
    parser.add_argument("--source", "-s", type=str, help="Путь к видеофайлу")
    parser.add_argument("--no-half", action="store_true", help="Отключить half-precision (FP16)")

    args = parser.parse_args()

    # Обязательная проверка CUDA
    if not torch.cuda.is_available():
        raise SystemExit("CUDA не обнаружена. Скрипт поддерживает только запуск на GPU (CUDA).")

    # Если аргументы не заданы — спросим интерактивно (но без «лишнего» функционала)
    if args.version is None:
        print("Выберите версию YOLO (5/6/7/8/9/10/11/12): ", end="", flush=True)
        try:
            args.version = int(input().strip())
        except Exception:
            raise SystemExit("Неверный ввод версии.")

    if not args.weights:
        print("Укажите путь к .pt весам: ", end="", flush=True)
        args.weights = input().strip()

    if not args.source:
        print("Укажите путь к видеофайлу: ", end="", flush=True)
        args.source = input().strip()

    # Нормализация путей
    weights_path = Path(args.weights).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()

    if not weights_path.exists():
        raise SystemExit(f"Файл весов не найден: {weights_path}")
    if not source_path.exists():
        raise SystemExit(f"Видеофайл не найден: {source_path}")

    device = "cuda:0"
    half = not args.no_half

    # Создаём детектор под выбранную версию
    detector = make_detector(args.version, str(weights_path), device=device, half=half)

    window_name = f"YOLOv{args.version} | {weights_path.name}"
    run(str(source_path), detector, window_name)


if __name__ == "__main__":
    main()
