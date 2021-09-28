from __future__ import print_function
import mxnet as mx
import numpy as np
import numpy.random as npr
from ..config import config, generate_config
from ..processing.generate_anchor import generate_anchors, anchors_plane
from ..processing.bbox_transform import bbox_overlaps, bbox_transform, landmark_transform

STAT = {0: 0}
STEP = 28800
DEBUG = False


class CascadeRefineOperator(mx.operator.CustomOp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(CascadeRefineOperator, self).__init__()
        self.stride = int(stride)
        self.prefix = prefix
        generate_config(network, dataset)
        self.mode = config.TRAIN.OHEM_MODE  # 0 for random 10:245, 1 for 10:246, 2 for 10:30, mode 1 for default
        stride = self.stride
        sstride = str(stride)
        base_size = config.RPN_ANCHOR_CFG[sstride]['BASE_SIZE']
        ratios = config.RPN_ANCHOR_CFG[sstride]['RATIOS']
        scales = config.RPN_ANCHOR_CFG[sstride]['SCALES']
        base_anchors = generate_anchors(base_size=base_size, ratios=list(ratios), scales=np.array(scales, dtype=np.float32),
                                        stride=stride, dense_anchor=config.DENSE_ANCHOR)
        num_anchors = base_anchors.shape[0]
        feat_height, feat_width = config.SCALES[0][0] // self.stride, config.SCALES[0][0] // self.stride
        feat_stride = self.stride

        A = num_anchors
        K = feat_height * feat_width
        self.A = A

        all_anchors = anchors_plane(feat_height, feat_width, feat_stride, base_anchors)
        all_anchors = all_anchors.reshape((K * A, 4))
        self.ori_anchors = all_anchors
        self.nbatch = 0
        global STAT
        for k in config.RPN_FEAT_STRIDE:
            STAT[k] = [0, 0, 0]

    def apply_bbox_pred(self, bbox_pred, ind=None):
        box_deltas = bbox_pred
        box_deltas[:, 0::4] = box_deltas[:, 0::4] * config.TRAIN.BBOX_STDS[0]
        box_deltas[:, 1::4] = box_deltas[:, 1::4] * config.TRAIN.BBOX_STDS[1]
        box_deltas[:, 2::4] = box_deltas[:, 2::4] * config.TRAIN.BBOX_STDS[2]
        box_deltas[:, 3::4] = box_deltas[:, 3::4] * config.TRAIN.BBOX_STDS[3]
        if ind is None:
            boxes = self.ori_anchors
        else:
            boxes = self.ori_anchors[ind]

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        # x1
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        # y1
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        # x2
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        # y2
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)
        return pred_boxes

    def assign_anchor_fpn(self, gt_label, anchors, landmark=False, prefix='face'):
        IOU = config.TRAIN.CASCADE_OVERLAP

        gt_boxes = gt_label['gt_boxes']
        if landmark:
            gt_landmarks = gt_label['gt_landmarks']
            assert gt_boxes.shape[0] == gt_landmarks.shape[0]
        bbox_pred_len = 4
        landmark_pred_len = 10
        num_anchors = anchors.shape[0]
        A = self.A
        feat_height, feat_width = config.SCALES[0][0] // self.stride, config.SCALES[0][0] // self.stride

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((num_anchors,), dtype=np.float32)
        labels.fill(-1)

        if gt_boxes.size > 0:
            # overlap between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(num_anchors), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            if not config.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < IOU[0]] = 0

            # fg label: for each gt, anchor with highest overlap
            if config.TRAIN.RPN_FORCE_POSITIVE:
                labels[gt_argmax_overlaps] = 1

            # fg label: above threshold IoU
            labels[max_overlaps >= IOU[1]] = 1

            if config.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < IOU[0]] = 0
        else:
            labels[:] = 0
        fg_inds = np.where(labels == 1)[0]

        # subsample positive labels if we have too many
        if config.TRAIN.RPN_ENABLE_OHEM == 0:
            fg_inds = np.where(labels == 1)[0]
            num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                if DEBUG:
                    disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
                labels[disable_inds] = -1

            # subsample negative labels if we have too many
            num_bg = config.TRAIN.RPN_BATCH_SIZE - np.sum(labels == 1)
            bg_inds = np.where(labels == 0)[0]
            if len(bg_inds) > num_bg:
                disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                if DEBUG:
                    disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
                labels[disable_inds] = -1
        else:
            fg_inds = np.where(labels == 1)[0]
            num_fg = len(fg_inds)
            bg_inds = np.where(labels == 0)[0]
            num_bg = len(bg_inds)

        bbox_targets = np.zeros((num_anchors, bbox_pred_len), dtype=np.float32)
        if gt_boxes.size > 0:
            bbox_targets[:, :] = bbox_transform(anchors, gt_boxes[argmax_overlaps, :])

        bbox_weights = np.zeros((num_anchors, bbox_pred_len), dtype=np.float32)
        bbox_weights[labels == 1, 0:4] = 1.0
        if bbox_pred_len > 4:
            bbox_weights[labels == 1, 4:bbox_pred_len] = 0.1

        if landmark:
            landmark_targets = np.zeros((num_anchors, landmark_pred_len), dtype=np.float32)
            landmark_weights = np.zeros((num_anchors, landmark_pred_len), dtype=np.float32)
            if landmark_pred_len == 10:
                landmark_weights[labels == 1, :] = 1.0
            elif landmark_pred_len == 15:
                v = [1.0, 1.0, 0.1] * 5
                assert len(v) == 15
                landmark_weights[labels == 1, :] = np.array(v)
            else:
                assert False
            if gt_landmarks.size > 0:
                a_landmarks = gt_landmarks[argmax_overlaps, :, :]
                landmark_targets[:] = landmark_transform(anchors, a_landmarks)
                invalid = np.where(a_landmarks[:, 0, 2] < 0.0)[0]
                landmark_weights[invalid, :] = 0.0

        bbox_targets[:, 0::4] = bbox_targets[:, 0::4] / config.TRAIN.BBOX_STDS[0]
        bbox_targets[:, 1::4] = bbox_targets[:, 1::4] / config.TRAIN.BBOX_STDS[1]
        bbox_targets[:, 2::4] = bbox_targets[:, 2::4] / config.TRAIN.BBOX_STDS[2]
        bbox_targets[:, 3::4] = bbox_targets[:, 3::4] / config.TRAIN.BBOX_STDS[3]

        label = {}
        _label = labels.reshape((1, feat_height, feat_width, A)).transpose(0, 3, 1, 2)
        _label = _label.reshape((1, A * feat_height * feat_width))
        bbox_target = bbox_targets.reshape((1, feat_height * feat_width, A * bbox_pred_len)).transpose(0, 2, 1)
        bbox_weight = bbox_weights.reshape((1, feat_height * feat_width, A * bbox_pred_len)).transpose((0, 2, 1))
        label['%s_label' % prefix] = _label[0]
        label['%s_bbox_target' % prefix] = bbox_target[0]
        label['%s_bbox_weight' % prefix] = bbox_weight[0]

        return label

    def forward(self, is_train, req, in_data, out_data, aux):
        self.nbatch += 1
        global STAT
        A = config.NUM_ANCHORS

        cls_score = in_data[2].asnumpy()  # BS, C, AHW
        bbox_pred_t0 = in_data[3].asnumpy()  # BS, AC, HW
        bbox_target_t0 = in_data[4].asnumpy()  # BS, AC, HW
        gt_boxes = in_data[6].asnumpy()  # BS, N, C=4+1

        batch_size = cls_score.shape[0]
        num_anchors = cls_score.shape[2]

        labels_out = np.zeros(shape=(batch_size, num_anchors), dtype=np.float32)
        bbox_target_out = np.zeros(shape=bbox_target_t0.shape, dtype=np.float32)
        anchor_weight = np.zeros((batch_size, num_anchors, 1), dtype=np.float32)
        valid_count = np.zeros((batch_size, 1), dtype=np.float32)

        bbox_pred_t0 = bbox_pred_t0.transpose((0, 2, 1))
        bbox_pred_t0 = bbox_pred_t0.reshape((bbox_pred_t0.shape[0], -1, 4))  # BS, H*W*A, C
        bbox_target_t0 = bbox_target_t0.transpose((0, 2, 1))
        bbox_target_t0 = bbox_target_t0.reshape((bbox_target_t0.shape[0], -1, 4))

        FAST = False
        for ibatch in range(batch_size):
            if not FAST:
                _gt_boxes = gt_boxes[ibatch]  # N, 4+1
                _gtind = len(np.where(_gt_boxes[:, 4] >= 0)[0])
                _gt_boxes = _gt_boxes[0:_gtind, :]

                anchors_t1 = self.apply_bbox_pred(bbox_pred_t0[ibatch])
                assert anchors_t1.shape[0] == self.ori_anchors.shape[0]

                gt_label = {'gt_boxes': _gt_boxes}
                new_label_dict = self.assign_anchor_fpn(gt_label, anchors_t1, False, prefix=self.prefix)
                labels = new_label_dict['%s_label' % self.prefix]  # AHW
                new_bbox_target = new_label_dict['%s_bbox_target' % self.prefix]  # AC,HW
                _anchor_weight = np.zeros((num_anchors, 1), dtype=np.float32)
                fg_score = cls_score[ibatch, 1, :] - cls_score[ibatch, 0, :]
                fg_inds = np.where(labels > 0)[0]
                num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
                if len(fg_inds) > num_fg:
                    if self.mode == 0:
                        disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                        labels[disable_inds] = -1
                    else:
                        pos_ohem_scores = fg_score[fg_inds]
                        order_pos_ohem_scores = pos_ohem_scores.ravel().argsort()
                        sampled_inds = fg_inds[order_pos_ohem_scores[:num_fg]]
                        labels[fg_inds] = -1
                        labels[sampled_inds] = 1

                n_fg = np.sum(labels > 0)
                fg_inds = np.where(labels > 0)[0]
                num_bg = config.TRAIN.RPN_BATCH_SIZE - n_fg
                if self.mode == 2:
                    num_bg = max(48, n_fg * int(1.0 / config.TRAIN.RPN_FG_FRACTION - 1))

                bg_inds = np.where(labels == 0)[0]
                if num_bg == 0:
                    labels[bg_inds] = -1
                elif len(bg_inds) > num_bg:
                    # sort ohem scores
                    if self.mode == 0:
                        disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                        labels[disable_inds] = -1
                    else:
                        neg_ohem_scores = fg_score[bg_inds]
                        order_neg_ohem_scores = neg_ohem_scores.ravel().argsort()[::-1]
                        sampled_inds = bg_inds[order_neg_ohem_scores[:num_bg]]
                        labels[bg_inds] = -1
                        labels[sampled_inds] = 0

                if n_fg > 0:
                    order0_labels = labels.reshape((1, A, -1)).transpose((0, 2, 1)).reshape((-1,))
                    bbox_fg_inds = np.where(order0_labels > 0)[0]
                    _anchor_weight[bbox_fg_inds, :] = 1.0
                anchor_weight[ibatch] = _anchor_weight
                valid_count[ibatch][0] = n_fg
                labels_out[ibatch] = labels
                bbox_target_out[ibatch] = new_bbox_target

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('cascade_refine')
class CascadeRefineProp(mx.operator.CustomOpProp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(CascadeRefineProp, self).__init__(need_top_grad=False)
        self.stride = stride
        self.network = network
        self.dataset = dataset
        self.prefix = prefix

    def list_arguments(self):
        return ['cls_label_t0', 'cls_pred_t0', 'cls_pred', 'bbox_pred_t0', 'bbox_label_t0', 'cls_label_raw', 'cas_gt_boxes']

    def list_outputs(self):
        return ['cls_label_out', 'bbox_label_out', 'anchor_weight_out', 'pos_count_out']

    def infer_shape(self, in_shape):
        cls_pred_shape = in_shape[1]
        bs = cls_pred_shape[0]
        num_anchors = cls_pred_shape[2]
        cls_label_shape = [bs, num_anchors]

        return in_shape, [cls_label_shape, in_shape[4], [bs, num_anchors, 1], [bs, 1]]

    def create_operator(self, ctx, shapes, dtypes):
        return CascadeRefineOperator(self.stride, self.network, self.dataset, self.prefix)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
