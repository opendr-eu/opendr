# MIT License
#
# Copyright (c) 2018 Jiankang Deng and Jia Guo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import datetime
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import cv2

from opendr.perception.object_detection_2d.retinaface.algorithm.processing.generate_anchor import generate_anchors_fpn, \
    anchors_plane
from opendr.perception.object_detection_2d.retinaface.algorithm.processing.bbox_transform import clip_boxes
from opendr.perception.object_detection_2d.retinaface.algorithm.processing.nms import cpu_nms_wrapper, gpu_nms_wrapper


class RetinaFace:
    def __init__(self, prefix=None, epoch=0, ctx_id=0, network='net3', nms=0.4, nocrop=False, decay4=0.5, vote=False,
                 sym=None, arg_params=None, aux_params=None, model=None):
        if prefix is None:
            assert (sym is not None and arg_params is not None and aux_params is not None) or model is not None, \
                    "If prefix is not set, (sym, arg_params and aux_params) or model must be provided."
        self.ctx_id = ctx_id
        self.network = network
        self.decay4 = decay4
        self.nms_threshold = nms
        self.vote = vote
        self.nocrop = nocrop
        self.debug = False
        self.fpn_keys = []
        self.anchor_cfg = None
        pixel_means = [0.0, 0.0, 0.0]
        pixel_stds = [1.0, 1.0, 1.0]
        pixel_scale = 1.0
        self.preprocess = False
        self.cov = False
        _ratio = (1.,)
        fmc = 3
        if network == 'ssh' or network == 'vgg':
            pixel_means = [103.939, 116.779, 123.68]
            self.preprocess = True
        elif network == 'net3':
            _ratio = (1.,)
        elif network == 'net3l':
            _ratio = (1.,)
            self.landmark_std = 0.2
            self.cov = True
            print('Detecting masked faces...')
        elif network == 'net3a':
            _ratio = (1., 1.5)
        elif network == 'net6':  # like pyramidbox or s3fd
            fmc = 6
        elif network == 'net5':  # retinaface
            fmc = 5
        elif network == 'net5a':
            fmc = 5
            _ratio = (1., 1.5)
        elif network == 'net4':
            fmc = 4
        elif network == 'net4a':
            fmc = 4
            _ratio = (1., 1.5)
        elif network == 'x5':
            fmc = 5
            pixel_means = [103.52, 116.28, 123.675]
            pixel_stds = [57.375, 57.12, 58.395]
        elif network == 'x3':
            fmc = 3
            pixel_means = [103.52, 116.28, 123.675]
            pixel_stds = [57.375, 57.12, 58.395]
        elif network == 'x3a':
            fmc = 3
            _ratio = (1., 1.5)
            pixel_means = [103.52, 116.28, 123.675]
            pixel_stds = [57.375, 57.12, 58.395]
        else:
            assert False, 'network setting error %s' % network

        if fmc == 3:
            self._feat_stride_fpn = [32, 16, 8]
            self.anchor_cfg = {
                '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        elif fmc == 4:
            self._feat_stride_fpn = [32, 16, 8, 4]
            self.anchor_cfg = {
                '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '4': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        elif fmc == 6:
            self._feat_stride_fpn = [128, 64, 32, 16, 8, 4]
            self.anchor_cfg = {
                '128': {'SCALES': (32,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '64': {'SCALES': (16,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '32': {'SCALES': (8,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (4,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '4': {'SCALES': (1,), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        elif fmc == 5:
            self._feat_stride_fpn = [64, 32, 16, 8, 4]
            self.anchor_cfg = {}
            _ass = 2.0 ** (1.0 / 3)
            _basescale = 1.0
            for _stride in [4, 8, 16, 32, 64]:
                key = str(_stride)
                value = {'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999}
                scales = []
                for _ in range(3):
                    scales.append(_basescale)
                    _basescale *= _ass
                value['SCALES'] = tuple(scales)
                self.anchor_cfg[key] = value

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        dense_anchor = False
        self._anchors_fpn = dict(
            zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=dense_anchor, cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v

        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))

        if prefix is not None:
            sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        elif sym is None:
            sym = model.symbol
            arg_params, aux_params = model.get_params()
        if self.ctx_id >= 0:
            self.ctx = mx.gpu(self.ctx_id)
            self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
        else:
            self.ctx = mx.cpu()
            self.nms = cpu_nms_wrapper(self.nms_threshold)
        self.pixel_means = np.array(pixel_means, dtype=np.float32)
        self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
        self.pixel_scale = float(pixel_scale)

        self.use_landmarks = False
        if len(sym) // len(self._feat_stride_fpn) >= 3:
            self.use_landmarks = True

        self.cascade = 0
        if float(len(sym)) // len(self._feat_stride_fpn) > 3.0:
            self.cascade = 1

        self.bbox_stds = [1.0, 1.0, 1.0, 1.0]
        self.landmark_std = 1.0
        if self.cov:
            self.landmark_std = 0.2
            self.cascade = False
            self.use_landmarks = True
            self.vote = False

        if self.debug:
            c = len(sym) // len(self._feat_stride_fpn)
            sym = sym[(c * 0):]
            self._feat_stride_fpn = [32, 16, 8]

        image_size = (640, 640)
        if model is None:
            self.model = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
            self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
            self.model.set_params(arg_params, aux_params)
        else:
            self.model = model
            self.model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], for_training=False)
            self.model.set_params(arg_params, aux_params)

    def detect(self, img, threshold=0.5, scales=[1.0], do_flip=False):
        proposals_list = []
        scores_list = []
        mask_scores_list = []
        landmarks_list = []
        strides_list = []
        timea = datetime.datetime.now()
        flips = [0]
        if do_flip:
            flips = [0, 1]

        imgs = [img]
        if isinstance(img, list):
            imgs = img
        for img in imgs:
            if isinstance(img, mx.ndarray.NDArray):
                img = img.asnumpy()
            for im_scale in scales:
                for flip in flips:
                    if im_scale != 1.0:
                        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                    else:
                        im = img.copy()
                    if flip:
                        im = im[:, ::-1, :]
                    if self.nocrop:
                        if im.shape[0] % 32 == 0:
                            h = im.shape[0]
                        else:
                            h = (im.shape[0] // 32 + 1) * 32
                        if im.shape[1] % 32 == 0:
                            w = im.shape[1]
                        else:
                            w = (im.shape[1] // 32 + 1) * 32
                        _im = np.zeros((h, w, 3), dtype=np.float32)
                        _im[0:im.shape[0], 0:im.shape[1], :] = im
                        im = _im
                    else:
                        im = im.astype(np.float32)
                    if self.debug:
                        time_b = datetime.datetime.now()
                        diff = time_b - timea
                        print('X1 uses', diff.total_seconds(), 'seconds')
                    im_info = [im.shape[0], im.shape[1]]
                    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
                    for i in range(3):
                        im_tensor[0, i, :, :] = (im[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / \
                                                self.pixel_stds[2 - i]
                    if self.debug:
                        time_b = datetime.datetime.now()
                        diff = time_b - timea
                        print('X2 uses', diff.total_seconds(), 'seconds')
                    data = nd.array(im_tensor)
                    db = mx.io.DataBatch(data=(data,), provide_data=[('data', data.shape)])
                    if self.debug:
                        time_b = datetime.datetime.now()
                        diff = time_b - timea
                        print('X3 uses', diff.total_seconds(), 'seconds')
                    self.model.forward(db, is_train=False)
                    net_out = self.model.get_outputs()

                    sym_idx = 0

                    for _idx, s in enumerate(self._feat_stride_fpn):
                        # _key = 'stride%s' % s
                        stride = int(s)
                        is_cascade = False
                        if self.cascade:
                            is_cascade = True
                        scores = net_out[sym_idx].asnumpy()
                        if self.cov:
                            type_scores = net_out[sym_idx + 3].asnumpy()
                        if self.debug:
                            time_b = datetime.datetime.now()
                            diff = time_b - timea
                            print('A uses', diff.total_seconds(), 'seconds')
                        A = self._num_anchors['stride%s' % s]
                        scores = scores[:, A:, :, :]
                        if self.cov:
                            mask_scores = type_scores[:, A * 2:, :, :]  # x, A, x, x

                        bbox_deltas = net_out[sym_idx + 1].asnumpy()

                        height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

                        K = height * width
                        anchors_fpn = self._anchors_fpn['stride%s' % s]
                        anchors = anchors_plane(height, width, stride, anchors_fpn)
                        anchors = anchors.reshape((K * A, 4))
                        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                        if self.cov:
                            mask_scores = mask_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

                        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
                        bbox_pred_len = bbox_deltas.shape[3] // A
                        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
                        bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * self.bbox_stds[0]
                        bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * self.bbox_stds[1]
                        bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * self.bbox_stds[2]
                        bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * self.bbox_stds[3]
                        proposals = self.bbox_pred(anchors, bbox_deltas)

                        if is_cascade:
                            cascade_sym_num = 0
                            cls_cascade = False
                            bbox_cascade = False
                            __idx = [3, 4]
                            if not self.use_landmarks:
                                __idx = [2, 3]
                            for diff_idx in __idx:
                                if sym_idx + diff_idx >= len(net_out):
                                    break
                                body = net_out[sym_idx + diff_idx].asnumpy()
                                if body.shape[1] // A == 2:  # cls branch
                                    if cls_cascade or bbox_cascade:
                                        break
                                    else:
                                        cascade_scores = body[:, self._num_anchors['stride%s' % s]:, :, :]
                                        cascade_scores = cascade_scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
                                        scores = cascade_scores
                                        cascade_sym_num += 1
                                        cls_cascade = True
                                elif body.shape[1] // A == 4:  # bbox branch
                                    cascade_deltas = body.transpose((0, 2, 3, 1)).reshape((-1, bbox_pred_len))
                                    cascade_deltas[:, 0::4] = cascade_deltas[:, 0::4] * self.bbox_stds[0]
                                    cascade_deltas[:, 1::4] = cascade_deltas[:, 1::4] * self.bbox_stds[1]
                                    cascade_deltas[:, 2::4] = cascade_deltas[:, 2::4] * self.bbox_stds[2]
                                    cascade_deltas[:, 3::4] = cascade_deltas[:, 3::4] * self.bbox_stds[3]
                                    proposals = self.bbox_pred(proposals, cascade_deltas)
                                    cascade_sym_num += 1
                                    bbox_cascade = True

                        proposals = clip_boxes(proposals, im_info[:2])

                        if stride == 4 and self.decay4 < 1.0:
                            scores *= self.decay4

                        scores_ravel = scores.ravel()
                        order = np.where(scores_ravel >= threshold)[0]
                        proposals = proposals[order, :]
                        scores = scores[order]
                        if self.cov:
                            mask_scores = mask_scores[order]
                        if flip:
                            old_x1 = proposals[:, 0].copy()
                            old_x2 = proposals[:, 2].copy()
                            proposals[:, 0] = im.shape[1] - old_x2 - 1
                            proposals[:, 2] = im.shape[1] - old_x1 - 1

                        proposals[:, 0:4] /= im_scale

                        proposals_list.append(proposals)
                        scores_list.append(scores)
                        if self.cov:
                            mask_scores_list.append(mask_scores)

                        if self.nms_threshold < 0.0:
                            _strides = np.empty(shape=(scores.shape), dtype=np.float32)
                            _strides.fill(stride)
                            strides_list.append(_strides)

                        if not self.vote and self.use_landmarks:
                            landmark_deltas = net_out[sym_idx + 2].asnumpy()
                            landmark_pred_len = landmark_deltas.shape[1] // A
                            landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape(
                                (-1, 5, landmark_pred_len // 5))
                            landmark_deltas *= self.landmark_std
                            landmarks = self.landmark_pred(anchors, landmark_deltas)
                            landmarks = landmarks[order, :]

                            if flip:
                                landmarks[:, :, 0] = im.shape[1] - landmarks[:, :, 0] - 1
                                order = [1, 0, 2, 4, 3]
                                flandmarks = landmarks.copy()
                                for idx, a in enumerate(order):
                                    flandmarks[:, idx, :] = landmarks[:, a, :]
                                landmarks = flandmarks
                            landmarks[:, :, 0:2] /= im_scale
                            landmarks_list.append(landmarks)
                        if self.cov:
                            sym_idx += 4
                        else:
                            if self.use_landmarks:
                                sym_idx += 3
                            else:
                                sym_idx += 2
                            if is_cascade:
                                sym_idx += cascade_sym_num

        if self.debug:
            time_b = datetime.datetime.now()
            diff = time_b - timea
            print('B uses', diff.total_seconds(), 'seconds')
        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if self.use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            if self.nms_threshold < 0.0:
                return np.zeros((0, 6)), landmarks
            else:
                return np.zeros((0, 5)), landmarks
        scores = np.vstack(scores_list)
        if self.cov:
            mask_scores = np.vstack(mask_scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]
        if self.cov:
            mask_scores = mask_scores[order]
        if self.nms_threshold < 0.0:
            strides = np.vstack(strides_list)
            strides = strides[order]
        if not self.vote and self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        if self.nms_threshold > 0.0:
            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            if not self.vote:
                keep = self.nms(pre_det)
                if self.cov:
                    det = np.hstack((pre_det, mask_scores))
                else:
                    det = np.hstack((pre_det, proposals[:, 4:]))
                det = det[keep, :]
                if self.use_landmarks:
                    landmarks = landmarks[keep]
            else:
                det = np.hstack((pre_det, proposals[:, 4:]))
                det = self.bbox_vote(det)
        elif self.nms_threshold < 0.0:
            det = np.hstack((proposals[:, 0:4], scores, strides)).astype(np.float32, copy=False)
        else:
            det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)

        if self.debug:
            time_b = datetime.datetime.now()
            diff = time_b - timea
            print('C uses', diff.total_seconds(), 'seconds')
        return det, landmarks

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        """
        Transform the set of class-agnostic boxes into class-specific boxes
        by applying the predicted offsets (box_deltas)
        :param boxes: !important [N 4]
        :param box_deltas: [N, 4 * num_classes]
        :return: [N 4 * num_classes]
        """
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
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

        if box_deltas.shape[1] > 4:
            pred_boxes[:, 4:] = box_deltas[:, 4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
            pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y
        return pred

    def bbox_vote(self, det):
        if det.shape[0] == 0:
            return np.zeros((0, 5))
        dets = None
        while det.shape[0] > 0:
            if dets is not None and dets.shape[0] >= 750:
                break
            # IOU
            area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[0] + area[:] - inter)

            # nms
            merge_index = np.where(o >= self.nms_threshold)[0]
            det_accu = det[merge_index, :]
            det = np.delete(det, merge_index, 0)
            if merge_index.shape[0] <= 1:
                if det.shape[0] == 0:
                    try:
                        dets = np.row_stack((dets, det_accu))
                    except:
                        dets = det_accu
                continue
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                          axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            if dets is None:
                dets = det_accu_sum
            else:
                dets = np.row_stack((dets, det_accu_sum))
        dets = dets[0:750, :]
        return dets
