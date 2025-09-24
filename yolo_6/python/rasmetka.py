import os
import cv2
from tqdm import tqdm

# Если нужно экспортировать модель с другими размерами изображений (не как в датасете), иначе None
RESIZE_TO_IMGSZ = None # или None, если не нужно ресайзить


IMG_PATH = "dt_1/train/images"
files = os.listdir(IMG_PATH)

with open("data_subset.txt", "w") as fd:
  for i in tqdm(files):
    print(os.path.join(IMG_PATH, i), file=fd)
    if RESIZE_TO_IMGSZ:
      img = cv2.imread(os.path.join(IMG_PATH, i))
      img = cv2.resize(img, (RESIZE_TO_IMGSZ, RESIZE_TO_IMGSZ))
      cv2.imwrite(os.path.join(IMG_PATH, i), img)
