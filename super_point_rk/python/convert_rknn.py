# build_superpoint_rknn.py
import numpy as np
from rknn.api import RKNN

ONNX_PATH = '/home/robot/uni_yolo/superpoint_infer_engine/output/superpoint_240x320.onnx'
RKNN_PATH = 'superpoint_240x320.rknn'
DATASET_TXT = '/home/robot/uni_yolo/superpoint_infer_engine/images_re_320/dataset.txt'  # список путей к калибровочным изображениями (grayscale 480x640)

rknn = RKNN()

print('--> config')
# Модель ожидает вход в [0..1], а картинки чаще в [0..255]:
# std_values=255 эквивалентно делению на 255; mean=0; 1 канал.
rknn.config(
    target_platform='rk3588',
    mean_values=[[0]],
    std_values=[[255]],
    quantized_dtype='asymmetric_quantized-8',   # int8 квантизация
    optimization_level=3
)
print('done')

print('--> load onnx')
ret = rknn.load_onnx(model=ONNX_PATH)
assert ret == 0, 'load_onnx failed'
print('done')

print('--> build (INT8 quant)')
ret = rknn.build(do_quantization=True, dataset=DATASET_TXT)
assert ret == 0, 'build failed'
print('done')

print('--> export')
ret = rknn.export_rknn(RKNN_PATH)
assert ret == 0, 'export_rknn failed'
print('done')

rknn.release()
