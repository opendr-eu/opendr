# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from opendr.perception.object_detection_2d.detr.algorithm.util import box_ops
from opendr.perception.object_detection_2d.detr.algorithm.util.misc import (NestedTensor,
                                                                            nested_tensor_from_tensor_list,
                                                                            accuracy, get_world_size, interpolate,
                                                                            is_dist_avail_and_initialized)

from opendr.perception.object_detection_2d.detr.algorithm.models.backbone import build_backbone
from opendr.perception.object_detection_2d.gem.algorithm.models.backbone_mobilenetv2 import (build_backbone as
                                                                                             build_mobilenetv2_backbone)
from opendr.perception.object_detection_2d.detr.algorithm.models.matcher import build_matcher
from opendr.perception.object_detection_2d.detr.algorithm.models.segmentation import (DETRsegm, PostProcessPanoptic,
                                                                                      PostProcessSegm,
                                                                                      dice_loss, sigmoid_focal_loss)
from opendr.perception.object_detection_2d.detr.algorithm.models.transformer import build_transformer


class Weight_relu(nn.Module):
    def __init__(self, backbone_name):
        super(Weight_relu, self).__init__()

        if backbone_name == 'resnet50':
            self.conv1024out = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        elif backbone_name == 'mobilenetv2':
            self.conv1024out = nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=1)
        else:
            print("backbone_name can only be resnet50 or mobilenetv2 currently.")
        self.adapt_pool16x16 = nn.AdaptiveAvgPool2d((16, 16))
        self.conv256out = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.conv1out = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1)
        self.linearlayer = nn.Linear(256, 1)

        self._initialize_weights()

    def forward(self, x):
        src_1024 = F.relu(self.conv1024out(x))
        x1 = self.adapt_pool16x16(F.relu(self.conv256out(src_1024)))
        x1 = F.relu(self.conv1out(x1))
        x1 = x1.view(-1, self.num_flat_features(x1))
        x1 = F.relu(self.linearlayer(x1))
        x1 = x1.view(src_1024.shape[0], 1, 1, 1)

        return x1, src_1024

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def _initialize_weights(self):
        torch.manual_seed(45)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class sc_avg_detr(nn.Module):
    """
    This is the Scalar Average Multi-Modal DETR module that performs object detection
    """

    def __init__(self, backbone, backbone_name, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model  # 512 by default
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.backbone_ir = backbone
        self.aux_loss = aux_loss

        self.weight_rgb = Weight_relu(backbone_name)
        self.weight_ir = Weight_relu(backbone_name)

        print("Model Created")

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples[0], torch.Tensor):
            for i in range(0, len(samples)):
                samples[i] = nested_tensor_from_tensor_list(samples[i])

        features, pos = self.backbone(samples[0])
        features_ir, _ = self.backbone_ir(samples[1])

        src, mask = features[-1].decompose()
        src_ir, _ = features_ir[-1].decompose()

        x1, _ = self.weight_rgb(src)
        x2, _ = self.weight_ir(src_ir)

        with torch.no_grad():
            mean_src = torch.mean(src[:, 0:16, :, :])
            mean_src_ir = torch.mean(src_ir[:, 0:16, :, :])

        x1 = torch.mul(mean_src, x1)
        x2 = torch.mul(mean_src_ir, x2)

        averaged_features = (x1 * src + x2 * src_ir) / 2

        assert mask is not None
        hs = self.transformer(self.input_proj(averaged_features), mask, self.query_embed.weight, pos[-1])[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = _set_aux_loss(outputs_class, outputs_coord)
        out['auxiliary_test'] = (x1, x2)
        return out


class avg_baseline(nn.Module):
    """
    This is the Average Baseline of MultiModal DETR module that performs object detection
    """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model  # 512 by default
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.backbone_ir = backbone
        self.aux_loss = aux_loss
        print("Average Baseline Model Loaded")

    def forward(self, samples: NestedTensor):
        if isinstance(samples[0], torch.Tensor):
            for i in range(0, len(samples)):
                samples[i] = nested_tensor_from_tensor_list(samples[i])

        features, pos = self.backbone(samples[0])
        features_ir, _ = self.backbone_ir(samples[1])

        src, mask = features[-1].decompose()
        src_ir, _ = features_ir[-1].decompose()

        averaged_features = (src + src_ir) / 2

        assert mask is not None
        hs = self.transformer(self.input_proj(averaged_features), mask, self.query_embed.weight, pos[-1])[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = _set_aux_loss(outputs_class, outputs_coord)
        out['auxiliary_test'] = (1, 1)
        return out


@torch.jit.unused
def _set_aux_loss(outputs_class, outputs_coord):
    # this is a workaround to make torchscript happy, as torchscript
    # doesn't support dictionary with non-homogeneous values, such
    # as a dict having both a Tensor and a list.
    return [{'pred_logits': a, 'pred_boxes': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            # the following is the class error without taking into account the "no-object" class.
            # https://github.com/facebookresearch/detr/issues/41#issuecomment-638088527
            losses['class_error'] = 100 - accuracy(src_logits[idx][..., :-1], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args, fusion_method, backbone_name):
    # "You should always use num_classes = max_id + 1 where max_id is the highest class ID that you have in your
    # dataset." Reference: https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # device = torch.device(args.device)
    transformer = build_transformer(args)
    if fusion_method == 'sc_avg' and backbone_name == 'resnet50':
        backbone = build_backbone(args)
        model = sc_avg_detr(
            backbone,
            backbone_name,
            transformer,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )
    elif fusion_method == 'sc_avg' and backbone_name == 'mobilenetv2':
        backbone = build_mobilenetv2_backbone(args)
        model = sc_avg_detr(
            backbone,
            backbone_name,
            transformer,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )
    elif fusion_method == 'avg_baseline' and backbone_name == 'resnet50':
        backbone = build_backbone(args)
        model = avg_baseline(
            backbone,
            backbone_name,
            transformer,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    return model


def build_c(args):
    device = torch.device(args.device)

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(args.num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    return criterion


def build_pp(args):
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return postprocessors
