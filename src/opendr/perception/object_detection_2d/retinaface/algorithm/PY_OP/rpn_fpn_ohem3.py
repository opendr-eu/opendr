from __future__ import print_function
import mxnet as mx
import numpy as np
from ..config import config, generate_config

STAT = {0: 0}
STEP = 28800


class RPNFPNOHEM3Operator(mx.operator.CustomOp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(RPNFPNOHEM3Operator, self).__init__()
        self.stride = int(stride)
        self.prefix = prefix
        generate_config(network, dataset)
        self.mode = config.TRAIN.OHEM_MODE  # 0 for random 10:245, 1 for 10:246, 2 for 10:30, mode 1 for default
        global STAT
        for k in config.RPN_FEAT_STRIDE:
            STAT[k] = [0, 0, 0]

    def forward(self, is_train, req, in_data, out_data, aux):
        global STAT

        cls_score = in_data[0].asnumpy()  # BS, 2, ANCHORS
        labels_raw = in_data[1].asnumpy()  # BS, ANCHORS

        A = config.NUM_ANCHORS
        anchor_weight = np.zeros((labels_raw.shape[0], labels_raw.shape[1], 1), dtype=np.float32)
        valid_count = np.zeros((labels_raw.shape[0], 1), dtype=np.float32)

        # _stat = [0, 0, 0]
        for ibatch in range(labels_raw.shape[0]):
            _anchor_weight = np.zeros((labels_raw.shape[1], 1), dtype=np.float32)
            labels = labels_raw[ibatch]
            fg_score = cls_score[ibatch, 1, :] - cls_score[ibatch, 0, :]

            fg_inds = np.where(labels > 0)[0]
            num_fg = int(config.TRAIN.RPN_FG_FRACTION * config.TRAIN.RPN_BATCH_SIZE)
            # origin_num_fg = len(fg_inds)
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
            # origin_num_bg = len(bg_inds)
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

        for ind, val in enumerate([labels_raw, anchor_weight, valid_count]):
            val = mx.nd.array(val)
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('rpn_fpn_ohem3')
class RPNFPNOHEM3Prop(mx.operator.CustomOpProp):
    def __init__(self, stride=0, network='', dataset='', prefix=''):
        super(RPNFPNOHEM3Prop, self).__init__(need_top_grad=False)
        self.stride = stride
        self.network = network
        self.dataset = dataset
        self.prefix = prefix

    def list_arguments(self):
        return ['cls_score', 'labels']

    def list_outputs(self):
        return ['labels_ohem', 'anchor_weight', 'valid_count']

    def infer_shape(self, in_shape):
        labels_shape = in_shape[1]
        anchor_weight_shape = [labels_shape[0], labels_shape[1], 1]

        return in_shape, [labels_shape, anchor_weight_shape, [labels_shape[0], 1]]

    def create_operator(self, ctx, shapes, dtypes):
        return RPNFPNOHEM3Operator(self.stride, self.network, self.dataset, self.prefix)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
