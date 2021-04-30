import torch
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.torchplus_tanet.nn import one_hot
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.core import box_torch_ops
from enum import Enum

PI = 3.141592653589793


def compute_iou_loss(pred_reg, target_reg, cls_weights, thre, weights):
    """compute iou loss
    Args:
        pred_reg ([B,N,7] Tensor): normal boxes: x, y, z, w, l, h, r
        target_reg ([B,N,7] Tensor): anchors
    """
    cls_weights = torch.max(torch.sigmoid(cls_weights),
                            dim=-1)[0]
    pos = torch.sigmoid(cls_weights) > thre
    pos = pos & weights.type_as(pos)

    pos_cls_weights = cls_weights[pos]
    pos_idx = pos.unsqueeze(pos.dim()).expand_as(pred_reg)

    loc_p = pred_reg[pos_idx].view(-1, 7)
    loc_t = target_reg[pos_idx].view(-1, 7)

    if loc_p.shape[0] > 1.0:
        insect_x = torch.max(
            torch.min((loc_p[:, 0] + loc_p[:, 3] / 2),
                      (loc_t[:, 0] + loc_t[:, 3] / 2)) - torch.max(
                          (loc_p[:, 0] - loc_p[:, 3] / 2),
                          (loc_t[:, 0] - loc_t[:, 3] / 2)),
            pred_reg[:, 3].new().resize_(loc_p[:, 3].shape).fill_(1e-3),
        )
        insect_y = torch.max(
            torch.min((loc_p[:, 1] + loc_p[:, 4] / 2),
                      (loc_t[:, 1] + loc_t[:, 4] / 2)) - torch.max(
                          (loc_p[:, 1] - loc_p[:, 4] / 2),
                          (loc_t[:, 1] - loc_t[:, 4] / 2)),
            pred_reg[:, 4].new().resize_(loc_p[:, 4].shape).fill_(1e-3),
        )
        insect_z = torch.max(
            torch.min((loc_p[:, 2] + loc_p[:, 5] / 2),
                      (loc_t[:, 2] + loc_t[:, 5] / 2)) - torch.max(
                          (loc_p[:, 2] - loc_p[:, 5] / 2),
                          (loc_t[:, 2] - loc_t[:, 5] / 2)),
            pred_reg[:, 5].new().resize_(loc_p[:, 5].shape).fill_(1e-3),
        )

        insect_area = insect_x * insect_y * insect_z
        pred_area = torch.max(
            loc_p[:, 3] * loc_p[:, 4] * loc_p[:, 5],
            loc_p.new().resize_(loc_p[:, 3].shape).fill_(1e-3),
        )
        tar_area = loc_t[:, 3] * loc_t[:, 4] * loc_t[:, 5]
        iou_tmp = insect_area / (pred_area + tar_area - insect_area)
        iou_tmp = pos_cls_weights * iou_tmp

        iou_tmp = torch.max(iou_tmp,
                            iou_tmp.new().resize_(iou_tmp.shape).fill_(1e-4))
        iou_loss = -torch.log(iou_tmp)
        iou_loss = iou_loss.mean()
    else:
        iou_loss = cls_weights.mean().new().resize_(
            cls_weights.mean().shape).fill_(0.0)

    return iou_loss


def center_to_minmax_2d_torch(centers, dims, origin=0.5):
    return torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)


def limit_period_torch(val, offset=0.5, period=PI):
    return val - torch.floor(val / period + offset) * period


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[...,
                                                                       -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def rbbox2d_to_near_bbox_torch(rbboxes):
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = torch.abs(limit_period_torch(rots, 0.5, PI))
    cond = (rots_0_pi_div_2 > PI / 4).unsqueeze_(-1)  # [..., np.newaxis]
    bboxes_center = torch.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d_torch(bboxes_center[:, :2], bboxes_center[:,
                                                                           2:])
    return bboxes


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, 2] - box_a[:, 0]) *
         (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter))  # [A,B]
    area_b = (
        ((box_b[:, 2] - box_b[:, 0]) *
         (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def compare_torch(boxes1, boxes2):
    boxes1_bv = rbbox2d_to_near_bbox_torch(boxes1)
    boxes2_bv = rbbox2d_to_near_bbox_torch(boxes2)
    ret = jaccard(boxes1_bv, boxes2_bv)
    return ret


def similarity_fn_torch(anchors, gt_boxes):
    anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
    gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
    ret = compare_torch(anchors_rbv, gt_boxes_rbv)
    return ret


def huber_loss(error, delta):
    abs_error = torch.abs(error)
    delta = delta * torch.ones_like(abs_error)
    quadratic = torch.min(abs_error, delta)
    linear = abs_error - quadratic
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)


def corners_loss(batch_predict_boxes, batch_gt_boxes, weights):
    batch_size = batch_predict_boxes.shape[0]
    origin = [0.5, 1.0, 0.5]
    corner_loss_sum = 0
    for i in range(batch_size):
        predict_boxes = batch_predict_boxes[i, :, :]
        gt_boxes = batch_gt_boxes[i, :, :]
        mask = weights[i, :] > 0
        valid_predict_boxes = predict_boxes[mask, :]
        valid_gt_boxes = gt_boxes[mask, :]
        if valid_gt_boxes.shape[0] > 0:
            predict_box_corners = box_torch_ops.center_to_corner_box3d(
                valid_predict_boxes[:, :3],
                valid_predict_boxes[:, 3:6],
                valid_predict_boxes[:, 6],
                origin=origin,
                axis=2,
            )

            gt_box_corners = box_torch_ops.center_to_corner_box3d(
                valid_gt_boxes[:, :3],
                valid_gt_boxes[:, 3:6],
                valid_gt_boxes[:, 6],
                origin=origin,
                axis=2,
            )
            gt_box_corners_flip = box_torch_ops.center_to_corner_box3d(
                valid_gt_boxes[:, :3],
                valid_gt_boxes[:, 3:6],
                valid_gt_boxes[:, 6] + PI,
                origin=origin,
                axis=2,
            )

            corner_dist_ori = torch.sum(torch.norm(predict_box_corners -
                                                   gt_box_corners,
                                                   dim=-1),
                                        dim=-1)
            corner_dist_flip = torch.sum(torch.norm(predict_box_corners -
                                                    gt_box_corners_flip,
                                                    dim=-1),
                                         dim=-1)

            corner_dist = torch.min(corner_dist_ori, corner_dist_flip)

            corner_loss_sum += huber_loss(corner_dist, delta=1.0)

    return corner_loss_sum


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


def prepare_loss_weights(
    labels,
    pos_cls_weight=1.0,
    neg_cls_weight=1.0,
    loss_norm_type=LossNormType.NormByNumPositives,
    dtype=torch.float32,
):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def create_refine_loss(
    loc_loss_ftor,
    cls_loss_ftor,
    example,
    coarse_box_preds,
    coarse_cls_preds,
    refine_box_preds,
    refine_cls_preds,
    cls_targets,  # [B,H*W,1]
    cls_weights,  # [B,H*W]
    reg_targets,  # [B,H*W, 7]
    reg_weights,  # [B,H*W]
    num_class,
    encode_background_as_zeros=True,
    encode_rad_error_by_sin=True,
    box_code_size=7,
    reg_weights_ori=None,
):

    batch_size = example["anchors"].shape[0]
    anchors = example["anchors"].view(batch_size, -1, box_code_size)
    coarse_box_preds = coarse_box_preds.view(batch_size, -1, box_code_size)
    refine_box_preds = refine_box_preds.view(batch_size, -1, box_code_size)

    de_coarse_boxes = box_torch_ops.second_box_decode(coarse_box_preds,
                                                      anchors)

    de_gt_boxes = box_torch_ops.second_box_decode(reg_targets, anchors)
    new_gt = box_torch_ops.second_box_encode(de_gt_boxes, de_coarse_boxes)

    if encode_background_as_zeros:
        refine_conf = refine_cls_preds.view(batch_size, -1, num_class)

    else:
        refine_conf = refine_cls_preds.view(batch_size, -1, num_class + 1)

    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot(
        cls_targets,
        depth=num_class + 1,
        dtype=refine_box_preds.dtype)
    if encode_background_as_zeros:  # True
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(refine_box_preds, new_gt)
    refine_loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]    # [2,70400,7]

    refine_cls_losses = cls_loss_ftor(refine_conf,
                                      one_hot_targets,
                                      weights=cls_weights)  # [N, M]

    return (
        refine_loc_losses,
        refine_cls_losses,
    )


def create_refine_loss_V2(
    loc_loss_ftor,
    cls_loss_ftor,
    example,
    coarse_box_batch_preds,
    coarse_cls_batch_preds,
    refine_box_batch_preds,
    refine_cls_batch_preds,
    num_class,
    loss_norm_type,
    encode_background_as_zeros=True,
    encode_rad_error_by_sin=True,
    box_code_size=7,
):

    batch_size = example["anchors"].shape[0]
    batch_anchors_shape = example["anchors"].shape
    gt_batch_boxes = example["gt_boxes"]
    gt_batch_classes = example["gt_classes"]
    anchors_batch_mask = example["anchors_mask"]
    coarse_box_batch_preds = coarse_box_batch_preds.view(
        batch_size, -1, box_code_size)
    refine_box_batch_preds = refine_box_batch_preds.view(
        batch_size, -1, box_code_size)
    anchor_batch = example["anchors"].view(batch_size, -1, box_code_size)
    batch_out_True_bbox = torch.zeros(batch_anchors_shape,
                                      dtype=torch.float32).cuda()

    batch_out_True_label = -torch.ones(batch_anchors_shape[:2],
                                       dtype=torch.int64).cuda()
    for i in range(batch_size):

        anchors = anchor_batch[i, :, :]
        coarse_box_preds = coarse_box_batch_preds[i, :, :]

        de_coarse_boxes = box_torch_ops.second_box_decode(
            coarse_box_preds, anchors)
        anchors = de_coarse_boxes

        gt_boxes_mask = gt_batch_boxes[:, 0] == i
        gt_boxes = gt_batch_boxes[gt_boxes_mask, 1:]
        gt_classes = gt_batch_classes[gt_boxes_mask]

        anchors_mask = anchors_batch_mask[i, :]
        vaild_anchors = torch.arange(len(anchors_mask))[anchors_mask]

        num_inside = len(vaild_anchors)
        coarse_boxes = anchors[vaild_anchors, :]

        matched_threshold = 0.6 * torch.ones(num_inside).cuda()
        unmatched_threshold = 0.45 * torch.ones(num_inside).cuda()

        labels = -torch.ones((num_inside, ), dtype=torch.int64).cuda()
        gt_ids = -torch.ones((num_inside, ), dtype=torch.int64).cuda()
        if len(gt_boxes) > 0 and coarse_boxes.shape[0] > 0:
            # Compute overlaps between the anchors and the gt boxes overlaps
            anchor_by_gt_overlap = similarity_fn_torch(coarse_boxes, gt_boxes)

            # Map from anchor to gt box that has highest overlap
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(
                dim=1).type_as(labels)
            # For each anchor, amount of overlap with most overlapping gt box
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_inside),
                                                    anchor_to_gt_argmax]  #

            # Map from gt box to an anchor that has highest overlap
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(
                dim=0).type_as(labels)
            # For each gt box, amount of overlap with most overlapping anchor
            gt_to_anchor_max = anchor_by_gt_overlap[
                gt_to_anchor_argmax,
                torch.arange(anchor_by_gt_overlap.shape[1])]

            # must remove gt which doesn't match any anchor.
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            # Find all anchors that share the max overlap amount
            # (this includes many ties)
            mask = torch.eq(anchor_by_gt_overlap, gt_to_anchor_max)
            anchors_with_max_overlap = torch.argmax(mask, dim=0).sort()[0]
            anchors_with_max_overlap = anchors_with_max_overlap.type_as(labels)

            # Fg label: for each gt use anchors with highest overlap
            # (including ties)
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force
            # Fg label: above threshold IOU
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds]
            gt_ids[pos_inds] = gt_inds

            bg_inds = anchor_to_gt_max < unmatched_threshold
        else:
            bg_inds = torch.arange(num_inside)

        fg_inds = labels > 0

        if len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:

            labels[bg_inds] = 0
            # re-enable anchors_with_max_overlap
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        bbox_targets = torch.zeros((num_inside, box_code_size),
                                   dtype=coarse_boxes.dtype).cuda()

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            bbox_targets[fg_inds, :] = box_torch_ops.second_box_encode(
                gt_boxes[anchor_to_gt_argmax[fg_inds], :],
                coarse_boxes[fg_inds, :])

        bbox_outside_weights = torch.zeros((num_inside, ),
                                           dtype=coarse_boxes.dtype).cuda()
        bbox_outside_weights[labels > 0] = 1.0

        batch_out_True_bbox[i, vaild_anchors, :] = bbox_targets
        batch_out_True_label[i, vaild_anchors] = labels

    cls_weights, reg_weights, cared = prepare_loss_weights(
        batch_out_True_label,
        pos_cls_weight=1.0,  # pos_cls_weight = 1.0
        neg_cls_weight=1.0,  # neg_cls_weight = 1.0
        loss_norm_type=loss_norm_type,
        dtype=torch.float32,
    )
    cls_targets = batch_out_True_label * cared.type_as(batch_out_True_label)
    cls_targets = cls_targets.unsqueeze(-1)

    if encode_background_as_zeros:
        refine_conf = refine_cls_batch_preds.view(batch_size, -1, num_class)

    else:
        refine_conf = refine_cls_batch_preds.view(batch_size, -1,
                                                  num_class + 1)

    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot(
        cls_targets,
        depth=num_class + 1,
        dtype=refine_box_batch_preds.dtype)
    if encode_background_as_zeros:  # True
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(refine_box_batch_preds,
                                                    batch_out_True_bbox)
    refine_loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]    # [2,70400,7]

    refine_cls_losses = cls_loss_ftor(refine_conf,
                                      one_hot_targets,
                                      weights=cls_weights)  # [N, M]
    return refine_loc_losses, refine_cls_losses
