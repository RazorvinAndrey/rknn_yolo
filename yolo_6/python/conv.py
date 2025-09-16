#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, time
import torch
import torch.nn as nn
import onnx
from io import BytesIO

ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

from yolov6.utils.checkpoint import load_checkpoint
from yolov6.layers.common import RepVGGBlock
from yolov6.utils.events import LOGGER


def find_head_lists(model):
    """
    Ищем в модели головы списки регрессии и классов:
    head.reg_preds, head.cls_preds (обычно nn.ModuleList длины nl=3).
    Возвращаем (reg_preds_list, cls_preds_list, nl, nc).
    """
    def get_nc(m):
        for k in ('num_classes', 'nc', 'num_cls'):
            if hasattr(m, k):
                return int(getattr(m, k))
        return None

    for m in model.modules():
        if hasattr(m, 'reg_preds') and hasattr(m, 'cls_preds'):
            reg_preds = getattr(m, 'reg_preds')
            cls_preds = getattr(m, 'cls_preds')
            if isinstance(reg_preds, (nn.ModuleList, list)) and isinstance(cls_preds, (nn.ModuleList, list)):
                nl = len(reg_preds)
                nc = get_nc(m) or get_nc(model)
                return reg_preds, cls_preds, nl, int(nc) if nc is not None else None
    return None, None, None, None


class ExportWrapper(nn.Module):
    """
    Обёртка вокруг исходной модели: делает обычный forward,
    но возвращает девять нужных тензоров, собранных hook'ами.
    """
    def __init__(self, model, reg_preds, cls_preds, nl):
        super().__init__()
        self.model = model
        self.reg_preds = reg_preds
        self.cls_preds = cls_preds
        self.nl = nl

        self._reg_out = [None] * nl
        self._cls_out = [None] * nl
        self._hooks = []

        # навешиваем forward_hook на каждый reg_pred/cls_pred
        for i in range(nl):
            self._hooks.append(self.reg_preds[i].register_forward_hook(self._make_hook(self._reg_out, i)))
            self._hooks.append(self.cls_preds[i].register_forward_hook(self._make_hook(self._cls_out, i)))

    @staticmethod
    def _make_hook(storage, idx):
        def hook(module, inp, out):
            storage[idx] = out
        return hook

    def forward(self, x):
        # Сброс хранилищ
        for i in range(self.nl):
            self._reg_out[i] = None
            self._cls_out[i] = None

        # прогоняем исходную модель (её выход нам не нужен)
        _ = self.model(x)

        outs = []
        for i in range(self.nl):
            reg = self._reg_out[i]                  # (B,4,H,W)
            cls_logits = self._cls_out[i]           # (B,C,H,W)
            # на ONNX отдаём уже вероятности по классам
            cls = torch.sigmoid(cls_logits)
            cls_sum = torch.sum(cls, dim=1, keepdim=True)   # (B,1,H,W)
            cls_sum = torch.clamp(cls_sum, 0.0, 1.0)
            outs += [reg, cls, cls_sum]
        return tuple(outs)   # tuple, чтобы ONNX сделал отдельные выходы

    def cleanup(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='path to .pt')
    ap.add_argument('--img-size', nargs=2, type=int, default=[640, 640], help='h w')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--opset', type=int, default=19)
    ap.add_argument('--simplify', action='store_true')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()

    # 1) загрузка модели
    use_cuda = (args.device != 'cpu') and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else 'cpu')
    print(f'Loading checkpoint from {args.weights}')
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)
    print('Fusing model...')
    model.eval()

    # 2) RepVGG в deploy (если есть)
    for m in model.modules():
        if isinstance(m, RepVGGBlock):
            m.switch_to_deploy()

    # 3) находим списки reg_preds/cls_preds
    reg_preds, cls_preds, nl, nc = find_head_lists(model)
    assert reg_preds is not None and cls_preds is not None, \
        "Не нашли head.reg_preds/cls_preds в модели."
    if nl not in (3, 2):  # обычно 3, но пусть будет гибко
        print(f'[warn] необычное число уровней nl={nl}')
    print(f'Found head: nl={nl}, nc={nc if nc is not None else "?"}.')

    # 4) обёртка и dummy input
    wrapper = ExportWrapper(model, reg_preds, cls_preds, nl)
    dummy = torch.zeros(args.batch_size, 3, args.img_size[0], args.img_size[1], device=device)

    # 5) имена выходов
    scales = ('P3', 'P4', 'P5')[:nl]
    out_names = []
    for s in scales:
        out_names += [f'{s}_boxes', f'{s}_scores', f'{s}_scores_sum']

    # 6) dry run (важно, чтобы hooks наполнили буферы)
    _ = wrapper(dummy)

    # 7) экспорт в ONNX
    export_path = args.out or args.weights.replace('.pt', '_rknn.onnx')
    LOGGER.info('\nStarting to export ONNX...')
    with BytesIO() as f:
        torch.onnx.export(
            wrapper, dummy, f,
            input_names=['images'],
            output_names=out_names,
            opset_version=args.opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            dynamic_axes=None
        )
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)

    # 8) (опц.) упрощение
    if args.simplify:
        try:
            import onnxsim
            LOGGER.info('\nStarting to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check
        except Exception as e:
            print(f'[warn] onnx-simplifier failed: {e}')

    onnx.save(onnx_model, export_path)
    wrapper.cleanup()
    print(f'OK: saved {export_path}')
    print('Outputs:', ', '.join(out_names))
    print(f'Elapsed: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
