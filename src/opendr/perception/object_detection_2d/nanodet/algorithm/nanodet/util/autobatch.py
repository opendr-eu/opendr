# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch

from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.torch_utils import profile


def check_train_batch_size(model, imgsz=640, amp=True):
    # Check YOLOv5 training batch size
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size


def autobatch(model, imgsz=(640, 640), divisible=32, fraction=0.8, batch_size=16, batch_sizes=None):
    # Automatically estimate best YOLOv5 batch size to use `fraction` of available CUDA memory
    # Usage:
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))

    # Check device
    imgsz[0] = ((imgsz[0] + divisible - 1) // divisible) * divisible
    imgsz[1] = ((imgsz[1] + divisible - 1) // divisible) * divisible
    prefix = 'AutoBatch: '
    print(f'{prefix}Computing optimal batch size for --input_size {imgsz}')
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        print(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size
    if torch.backends.cudnn.benchmark:
        print(f'{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}')
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    print(f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')

    # Profile batch sizes
    if batch_sizes is None:
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    try:
        img = [torch.empty(b, 3, imgsz[0], imgsz[1]) for b in batch_sizes]
        results = profile(img, model, n=3, device=device, flops=False)
    except Exception as e:
        print(f'{prefix}{e}')

    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if (b < 1 or b > 1024) and (imgsz[0] * imgsz[1] < 102400):  # b outside of safe range
        # input smaller than 320*320 can go a lot further than 1024 batch size in modern hardware
        b = batch_sizes[-1]
        print(f'{prefix}WARNING ⚠️ CUDA anomaly detected using maximum batch size {b},'
              f' recommend restart environment and retry command.')
        return b
    elif (b < 1 or b > 1024):
        print(f'{prefix}WARNING ⚠️ CUDA anomaly detected,'
              f' recommend restart environment and retry command.')

    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    print(f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅')
    return b
