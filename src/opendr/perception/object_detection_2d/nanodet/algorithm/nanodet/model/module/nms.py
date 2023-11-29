import torch
from torchvision.ops import nms
from typing import Dict


def multiclass_nms(
    multi_bboxes,
    multi_scores,
    score_thr: float,
    nms_cfg: Dict[str, float],
    max_num: int = -1,
    score_factors: torch.Tensor = torch.empty(0)
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dictionary): dictionary of the type and threshold of IoU
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    if not (score_factors.numel() == 0):
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)

    # for scripting
    labels = torch.tensor(0, dtype=torch.long, device=valid_mask.device)
    torch.nonzero(valid_mask, out=labels)
    # labels = valid_mask.nonzero(as_tuple=False)#[:, 1]
    labels = labels[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels

    nms_cfg["nms_max_number"] = float(max_num)
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    return dets, labels[keep]


def batched_nms(boxes, scores, idxs, nms_cfg: Dict[str, float], class_agnostic: bool = False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - nms_max_number (float): int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    split_thr = nms_cfg_.pop("split_thr", 10000.0)
    if boxes_for_nms.shape[0] < split_thr:
        keep = nms(boxes_for_nms, scores, nms_cfg_["iou_threshold"])
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id)
            mask_out = torch.tensor(0, dtype=torch.long, device=boxes.device)
            torch.nonzero(mask, out=mask_out)
            mask = mask_out.view(-1)
            # mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = nms(boxes_for_nms[mask], scores[mask], nms_cfg_["iou_threshold"])
            total_mask[mask[keep]] = True

        keep_out = torch.tensor(0, dtype=torch.long, device=boxes.device)
        torch.nonzero(total_mask, out=keep_out)
        keep = keep_out.view(-1)
        # keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    dets = torch.cat([boxes, scores[:, None]], -1)
    max_num = int(nms_cfg_.pop("nms_max_number", 100.0))
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, keep
