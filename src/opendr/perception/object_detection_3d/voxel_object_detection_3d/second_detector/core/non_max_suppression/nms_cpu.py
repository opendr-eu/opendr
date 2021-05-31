from pathlib import Path

from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.utils.buildtools.pybind11_build import (
    load_pb11,
)
import numba
import numpy as np

try:
    from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.core.non_max_suppression.nms import (
        non_max_suppression_cpu,
        rotate_non_max_suppression_cpu,
    )
except:
    current_dir = Path(__file__).resolve().parents[0]
    load_pb11(
        # ["../cc/nms/nms_kernel.cu.cc", "../cc/nms/nms.cc"],
        ["../cc/nms/nms_cpu_only.cc"],
        current_dir / "nms.so",
        current_dir,
        # cuda=True,
        cuda=False,
    )
    from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.core.non_max_suppression.nms import (
        non_max_suppression_cpu,
        rotate_non_max_suppression_cpu,
    )

from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.core import (
    box_np_ops,
)


def nms_cc(dets, thresh):
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    return non_max_suppression_cpu(dets, order, thresh, 1.0)


def rotate_nms_cc(dets, thresh):

    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    dets_corners = box_np_ops.center_to_corner_box2d(
        dets[:, :2], dets[:, 2:4], dets[:, 4]
    )

    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)

    standup_iou = box_np_ops.iou_jit(dets_standup, dets_standup, eps=0.0)
    # print(dets_corners.shape, order.shape, standup_iou.shape)
    return rotate_non_max_suppression_cpu(dets_corners, order, standup_iou, thresh)


@numba.jit(nopython=True)
def nms_jit(dets, thresh, eps=0.0):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate iou between i and j box
            w = max(min(x2[i], x2[j]) - max(x1[i], x1[j]) + eps, 0.0)
            h = max(min(y2[i], y2[j]) - max(y1[i], y1[j]) + eps, 0.0)
            inter = w * h
            ovr = inter / (areas[i] + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep


@numba.jit("float32[:, :], float32, float32, float32, uint32", nopython=True)
def soft_nms_jit(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = min(tx2, x2) - max(tx1, x1) + 1
            if iw > 0:
                ih = min(ty2, y2) - max(ty1, y1) + 1
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N - 1, 0]
                        boxes[pos, 1] = boxes[N - 1, 1]
                        boxes[pos, 2] = boxes[N - 1, 2]
                        boxes[pos, 3] = boxes[N - 1, 3]
                        boxes[pos, 4] = boxes[N - 1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep
