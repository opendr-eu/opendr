# Copyright 2020-2021 OpenDR European Project
#
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

import os
import sys
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import mxnet as mx
from mxnet.module import Module
from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.engine.constants import OPENDR_SERVER_URL

from opendr.perception.object_detection_2d.retinaface.algorithm.models.retinaface import RetinaFace
from opendr.perception.object_detection_2d.retinaface.algorithm.utils.load_data import load_gt_roidb
from opendr.perception.object_detection_2d.retinaface.algorithm.core.loader import CropLoader
from opendr.perception.object_detection_2d.retinaface.algorithm.core import metric
from opendr.perception.object_detection_2d.retinaface.algorithm.config import config, generate_config
from opendr.perception.object_detection_2d.retinaface.algorithm.symbol.symbol_resnet import get_resnet_train
from opendr.perception.object_detection_2d.retinaface.algorithm.logger import logger
from opendr.perception.object_detection_2d.retinaface.algorithm.eval_recall import FaceDetectionRecallMetric
from opendr.perception.object_detection_2d.datasets.detection_dataset import DetectionDataset
from opendr.perception.object_detection_2d.datasets.wider_face import WiderFaceDataset
from opendr.perception.object_detection_2d.datasets.transforms import BoundingBoxListToNumpyArray


class RetinaFaceLearner(Learner):
    def __init__(self, backbone='resnet', lr=0.001, batch_size=2, checkpoint_after_iter=0, checkpoint_load_iter=0,
                 lr_steps='0', epochs=100, momentum=0.9, weight_decay=5e-4, log_after=20, prefix='',
                 shuffle=True, flip=False, val_after=5, temp_path='', device='cuda'):
        super(RetinaFaceLearner, self).__init__(lr=lr, batch_size=batch_size, backbone=backbone,
                                                checkpoint_after_iter=checkpoint_after_iter,
                                                checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                                device=device)
        self.device = device
        if device == 'cuda':
            if mx.context.num_gpus() > 0:
                self.gpu_id = 0
            else:
                self.gpu_id = -1
        else:
            # use cpu
            self.gpu_id = -1
        self.detector = None

        if self.backbone not in ['resnet', 'mnet']:
            raise ValueError("network must be one of ['resnet', 'mnet']")

        if self.backbone == 'resnet':
            self.net = 'net3'
        else:
            self.net = 'net3l'

        self.classes = ['face', 'masked face']

        self.flip = flip
        self.shuffle = shuffle
        self.lr_steps = [int(step) for step in lr_steps.split(',')]
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.log_after = log_after
        self.prefix = prefix
        self.val_after = val_after

    def __get_ctx(self):
        ctx = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ and self.device == 'cuda':
            cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
        elif self.device == 'cuda' and mx.context.num_gpus() > 0:
            cvd = ['0']
        else:
            cvd = []
        if len(cvd) > 0 and self.device == 'cuda':
            if isinstance(cvd, str):
                visibles_ids = cvd.split(',')
            elif isinstance(cvd, list):
                visibles_ids = cvd
            for i in visibles_ids:
                ctx.append(mx.gpu(int(i)))
        else:
            ctx = [mx.cpu()]
        return ctx

    def fit(self, dataset, val_dataset=None, from_scratch=False, silent=False, verbose=True):
        """
        This method is used to train the detector on the WIDER Face dataset. Validation if performed if a val_dataset is
        provided.
        :param dataset: training dataset object; only WiderFaceDataset is supported currently
        :type dataset: opendr.perception.object_detection_2d.datasets.WiderFaceDataset
        :param val_dataset: validation dataset object
        :type val_dataset: opendr.perception.object_detection_2d.datasets.DetectionDataset, optional
        :param from_scratch: indicates whether to train from scratch or to download and use a pretrained backbone
        :type from_scratch: bool, optional
        :param silent: if set to True, disables all printing to STDOUT, defaults to False
        :type silent: bool, optional
        :param verbose: if set to True, additional information is printed to STDOUT, defaults to True
        :type verbose: bool
        :return: returns stats regarding the training and validation process
        :rtype: dict
        """

        if silent:
            logger.setLevel(0)
            # verbose = False

        if self.backbone == "mnet":
            raise NotImplementedError("Only the 'resnet' backbone is supported for training")

        ctx = self.__get_ctx()
        input_batch_size = self.batch_size * len(ctx)

        checkpoint_path = os.path.join(self.temp_path, self.prefix)

        # prepare dataset for training, downloads extra annotations if they're not present in the dataset
        dataset = self.__prepare_dataset(dataset)
        self.eval_dataset = val_dataset

        # get roidbs
        image_sets = dataset.splits
        roidbs = [load_gt_roidb('retinaface',
                                'WIDER_' + image_set,
                                self.temp_path,
                                dataset.root, flip=self.flip,
                                verbose=verbose)
                  for image_set in image_sets]
        roidb = roidbs[0]
        generate_config(self.backbone, 'retinaface')

        start_epoch = 0
        # create network & get backbone weights
        sym = None
        if from_scratch:
            arg_params = {}
            aux_params = {}
        else:
            backbone_path = os.path.join(self.temp_path, "backbone")
            self.download(backbone_path, mode="backbone")
            backbone_prefix = os.path.join(backbone_path, "resnet-50")
            if self.checkpoint_load_iter > 0:
                if verbose:
                    print("Loading checkpoint from {}...".format(checkpoint_path))
                    mx.model.load_checkpoint(checkpoint_path, self.checkpoint_load_iter)
                    start_epoch = self.checkpoint_load_iter
            else:
                sym, arg_params, aux_params = mx.model.load_checkpoint(backbone_prefix, start_epoch)
        sym = get_resnet_train(sym)

        feat_sym = []
        for stride in config.RPN_FEAT_STRIDE:
            feat_sym.append(
                sym.get_internals()['face_rpn_cls_score_stride%s_output' % stride]
            )

        train_data = CropLoader(feat_sym, roidb, input_batch_size, shuffle=self.shuffle, ctx=ctx,
                                work_load_list=None)
        max_data_shape = [('data', (1, 3, max([v[1] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
        max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
        max_data_shape.append(('gt_boxes', (1, roidb[0]['max_num_boxes'], 5)))

        # infer shape
        data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
        arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
        arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
        # out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
        # aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))

        for k in arg_shape_dict:
            v = arg_shape_dict[k]
            if k.find('upsampling') >= 0:
                if verbose:
                    print('Initializing upsampling weight', k)
                arg_params[k] = mx.nd.zeros(shape=v)
                init = mx.init.Initializer()
                init._init_bilinear(k, arg_params[k])

        # fixed_param_prefix = config.FIXED_PARAMS
        # create solver
        data_names = [k[0] for k in train_data.provide_data]
        label_names = [k[0] for k in train_data.provide_label]
        fixed_param_names = self.__get_fixed_params(sym)
        if verbose and fixed_param_names:
            print('Fixed', fixed_param_names)

        mod = Module(sym,
                     data_names=data_names,
                     label_names=label_names,
                     context=ctx,
                     logger=logger,
                     fixed_param_names=fixed_param_names)
        self._model = mod

        eval_metrics = mx.metric.CompositeEvalMetric()
        train_dict = defaultdict(list)
        mid = 0
        for m in range(len(config.RPN_FEAT_STRIDE)):
            stride = config.RPN_FEAT_STRIDE[m]
            _metric = metric.RPNAccMetric(pred_idx=mid, label_idx=mid + 1, name='RPNAcc_s%s' % stride)
            eval_metrics.add(_metric)
            mid += 2

            _metric = metric.RPNL1LossMetric(loss_idx=mid, weight_idx=mid + 1, name='RPNL1Loss_s%s' % stride)
            eval_metrics.add(_metric)
            mid += 2

            if config.FACE_LANDMARK:
                _metric = metric.RPNL1LossMetric(loss_idx=mid, weight_idx=mid + 1, name='RPNLandMarkL1Loss_s%s' % stride)
                eval_metrics.add(_metric)
                mid += 2

            if config.HEAD_BOX:
                _metric = metric.RPNAccMetric(pred_idx=mid, label_idx=mid + 1, name='RPNAcc_head_s%s' % stride)
                eval_metrics.add(_metric)
                mid += 2

                _metric = metric.RPNL1LossMetric(loss_idx=mid, weight_idx=mid + 1, name='RPNL1Loss_head_s%s' % stride)
                eval_metrics.add(_metric)
                mid += 2

            if config.CASCADE > 0:
                for _idx in range(config.CASCADE):
                    if stride in config.CASCADE_CLS_STRIDES:
                        _metric = metric.RPNAccMetric(pred_idx=mid, label_idx=mid + 1, name='RPNAccCAS%d_s%s' % (_idx, stride))
                        eval_metrics.add(_metric)
                        mid += 2
                    if stride in config.CASCADE_BBOX_STRIDES:
                        _metric = metric.RPNL1LossMetric(loss_idx=mid, weight_idx=mid + 1, name='RPNL1LossCAS%d_s%s' % (_idx,
                                                                                                                        stride))
                        eval_metrics.add(_metric)
                        mid += 2

            # lr
            lr_epoch = [int(epoch) for epoch in self.lr_steps]
            lr_epoch_diff = [epoch - start_epoch for epoch in lr_epoch if epoch > start_epoch]
            lr_iters = [int(epoch * len(roidb)) / input_batch_size for epoch in lr_epoch_diff]
            iter_per_epoch = int(len(roidb) / input_batch_size)

            lr_steps = []
            if len(lr_iters) == 5:
                factors = [0.5, 0.5, 0.4, 0.1, 0.1]
                for i in range(5):
                    lr_steps.append((lr_iters[i], factors[i]))
            elif len(lr_iters) == 8:  # warmup
                for li in lr_iters[0:5]:
                    lr_steps.append((li, 1.5849))
                for li in lr_iters[5:]:
                    lr_steps.append((li, 0.1))
            else:
                for li in lr_iters:
                    lr_steps.append((li, 0.1))

            end_epoch = self.epochs

            opt = mx.optimizer.SGD(learning_rate=self.lr, momentum=self.momentum, wd=self.weight_decay,
                                   rescale_grad=1. / len(ctx), clip_gradient=None)
            initializer = mx.init.Xavier()
            train_data = mx.io.PrefetchingIter(train_data)
            _cb = mx.callback.Speedometer(train_data.batch_size, frequent=self.log_after, auto_reset=False)

            global_step = [0]

            def save_model(epoch):
                arg, aux = mod.get_params()
                all_layers = mod.symbol.get_internals()
                outs = []
                for stride in config.RPN_FEAT_STRIDE:
                    num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
                    if config.CASCADE > 0:
                        _name = 'face_rpn_cls_score_stride%d_output' % (stride)
                        cls_pred = all_layers[_name]
                        cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, 2, -1, 0))

                        cls_pred = mx.symbol.SoftmaxActivation(data=cls_pred, mode="channel")
                        cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, 2 * num_anchors, -1, 0))
                        outs.append(cls_pred)
                        _name = 'face_rpn_bbox_pred_stride%d_output' % stride
                        rpn_bbox_pred = all_layers[_name]
                        outs.append(rpn_bbox_pred)
                        if config.FACE_LANDMARK:
                            _name = 'face_rpn_landmark_pred_stride%d_output' % stride
                            rpn_landmark_pred = all_layers[_name]
                            outs.append(rpn_landmark_pred)
                        for casid in range(config.CASCADE):
                            if stride in config.CASCADE_CLS_STRIDES:
                                _name = 'face_rpn_cls_score_stride%d_cas%d_output' % (stride, casid)
                                cls_pred = all_layers[_name]
                                cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, 2, -1, 0))
                                cls_pred = mx.symbol.SoftmaxActivation(data=cls_pred, mode="channel")
                                cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, 2 * num_anchors, -1, 0))
                                outs.append(cls_pred)
                            if stride in config.CASCADE_BBOX_STRIDES:
                                _name = 'face_rpn_bbox_pred_stride%d_cas%d_output' % (stride, casid)
                                bbox_pred = all_layers[_name]
                                outs.append(bbox_pred)
                    else:
                        _name = 'face_rpn_cls_score_stride%d_output' % stride
                        rpn_cls_score = all_layers[_name]

                        # prepare rpn data
                        rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                                  shape=(0, 2, -1, 0),
                                                                  name="face_rpn_cls_score_reshape_stride%d" % stride)

                        rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                                   mode="channel",
                                                                   name="face_rpn_cls_prob_stride%d" % stride)
                        rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                                 name='face_rpn_cls_prob_reshape_stride%d' % stride)
                        _name = 'face_rpn_bbox_pred_stride%d_output' % stride
                        rpn_bbox_pred = all_layers[_name]
                        outs.append(rpn_cls_prob_reshape)
                        outs.append(rpn_bbox_pred)
                        if config.FACE_LANDMARK:
                            _name = 'face_rpn_landmark_pred_stride%d_output' % stride
                            rpn_landmark_pred = all_layers[_name]
                            outs.append(rpn_landmark_pred)
                _sym = mx.sym.Group(outs)
                mx.model.save_checkpoint(checkpoint_path, epoch, _sym, arg, aux)

            def _batch_callback(param):
                # global global_step
                _cb(param)
                global_step[0] += 1
                mbatch = global_step[0]
                for step in lr_steps:
                    if mbatch == step[0]:
                        opt.lr *= step[1]
                        if verbose:
                            print('lr change to', opt.lr, ' in batch', mbatch, file=sys.stderr)
                        break

                if self.checkpoint_after_iter > 0 and mbatch % iter_per_epoch == self.checkpoint_after_iter - 1:
                    metrics = param.eval_metric.metrics
                    for m in metrics:
                        ks, vals = m.get()
                        if isinstance(ks, list):
                            for m_idx, k in enumerate(ks):
                                train_dict[k].append(vals[m_idx])
                        else:
                            train_dict[ks].append(vals)

                    save_model(int((mbatch - 1) / iter_per_epoch))

            # train:
            mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=self.__epoch_callback,
                    batch_end_callback=_batch_callback, kvstore='device',
                    optimizer=opt,
                    initializer=initializer,
                    allow_missing=True,
                    arg_params=arg_params, aux_params=aux_params, begin_epoch=start_epoch, num_epoch=end_epoch)

            if verbose:
                for k, v in train_dict.items():
                    print(k, len(v), v[0], v[-1])
            return train_dict

    @staticmethod
    def __prepare_detector(mod):
        """
        This method makes some necessary modifications to the model in order to prepare it for inference.
        """
        arg, aux = mod.get_params()
        all_layers = mod.symbol.get_internals()
        outs = []
        for stride in config.RPN_FEAT_STRIDE:
            num_anchors = config.RPN_ANCHOR_CFG[str(stride)]['NUM_ANCHORS']
            if config.CASCADE > 0:
                _name = 'face_rpn_cls_score_stride%d_output' % (stride)
                cls_pred = all_layers[_name]
                cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, 2, -1, 0))

                cls_pred = mx.symbol.SoftmaxActivation(data=cls_pred, mode="channel")
                cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, 2 * num_anchors, -1, 0))
                outs.append(cls_pred)
                _name = 'face_rpn_bbox_pred_stride%d_output' % stride
                rpn_bbox_pred = all_layers[_name]
                outs.append(rpn_bbox_pred)
                if config.FACE_LANDMARK:
                    _name = 'face_rpn_landmark_pred_stride%d_output' % stride
                    rpn_landmark_pred = all_layers[_name]
                    outs.append(rpn_landmark_pred)
                for casid in range(config.CASCADE):
                    if stride in config.CASCADE_CLS_STRIDES:
                        _name = 'face_rpn_cls_score_stride%d_cas%d_output' % (stride, casid)
                        cls_pred = all_layers[_name]
                        cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, 2, -1, 0))
                        cls_pred = mx.symbol.SoftmaxActivation(data=cls_pred, mode="channel")
                        cls_pred = mx.symbol.Reshape(data=cls_pred, shape=(0, 2 * num_anchors, -1, 0))
                        outs.append(cls_pred)
                    if stride in config.CASCADE_BBOX_STRIDES:
                        _name = 'face_rpn_bbox_pred_stride%d_cas%d_output' % (stride, casid)
                        bbox_pred = all_layers[_name]
                        outs.append(bbox_pred)
            else:
                _name = 'face_rpn_cls_score_stride%d_output' % stride
                rpn_cls_score = all_layers[_name]

                # prepare rpn data
                rpn_cls_score_reshape = mx.symbol.Reshape(data=rpn_cls_score,
                                                          shape=(0, 2, -1, 0),
                                                          name="face_rpn_cls_score_reshape_stride%d" % stride)

                rpn_cls_prob = mx.symbol.SoftmaxActivation(data=rpn_cls_score_reshape,
                                                           mode="channel",
                                                           name="face_rpn_cls_prob_stride%d" % stride)
                rpn_cls_prob_reshape = mx.symbol.Reshape(data=rpn_cls_prob,
                                                         shape=(0, 2 * num_anchors, -1, 0),
                                                         name='face_rpn_cls_prob_reshape_stride%d' % stride)
                _name = 'face_rpn_bbox_pred_stride%d_output' % stride
                rpn_bbox_pred = all_layers[_name]
                outs.append(rpn_cls_prob_reshape)
                outs.append(rpn_bbox_pred)
                if config.FACE_LANDMARK:
                    _name = 'face_rpn_landmark_pred_stride%d_output' % stride
                    rpn_landmark_pred = all_layers[_name]
                    outs.append(rpn_landmark_pred)
        _sym = mx.sym.Group(outs)
        return _sym, arg, aux

    def __epoch_callback(self, epoch, symbol, arg_params, aux_params):
        """
        Callback method, called at the end of each training epoch. Evaluation is performed if a validation dataset has been
        provided by the user, every 'val_after' epochs.
        """
        if epoch % self.val_after == self.val_after - 1 and self.eval_dataset is not None:
            sym, arg, aux = self.__prepare_detector(self._model)
            self.detector = RetinaFace(network=self.net, sym=sym, arg_params=arg, aux_params=aux, model=None)
            self.eval(self.eval_dataset, use_subset=True, subset_size=500, pyramid=False, flip=False)

    def eval(self, dataset, verbose=True, use_subset=False, subset_size=250, pyramid=True, flip=True):
        """
        This method performs evaluation on a given dataset and returns a dictionary with the evaluation results.
        :param dataset: dataset object, to perform evaluation on
        :type dataset: opendr.perception.object_detection_2d.datasets.DetectionDataset
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional
        :param use_subset: if True, only a subset of the dataset is evaluated, defaults to False
        :type use_subset: bool, optional
        :param subset_size: if use_subset is True, subset_size controls the size of the subset to be evaluated
        :type subset_size: int, optional
        :param pyramid: if True, an image pyramid is used during evaluation to increase performance
        :type pyramid: bool, optional
        :param flip: if True, images are flipped during evaluation to increase performance
        :type flip: bool, optional
        :return: dictionary containing evaluation metric names nad values
        :rtype: dict
        """
        if self.detector is None:
            assert "Detector must be loaded with load() before inference."

        if verbose:
            print("Evaluation params: [pyramid={}, flip={}]".format(pyramid, flip))

        # prepare dataset & get metric
        dataset, eval_metric = self.__prepare_val_dataset(dataset)

        if use_subset:
            val_indices = np.random.choice(range(len(dataset)), subset_size)
            eval_metric.n_val_images = subset_size
            if verbose:
                print("Using random subset of {} images...".format(subset_size))
        else:
            val_indices = np.arange(0, len(dataset))

        # perform evaluation
        eval_metric.reset()
        for idx in tqdm(val_indices):
            img, labels = dataset[idx]
            if isinstance(img, Image):
                img = img.data
            if isinstance(labels, BoundingBoxList):
                labels = BoundingBoxListToNumpyArray()(labels)
            do_flip = flip

            if not pyramid:
                # target_size = 1600
                target_size = 640
                # max_size = 2150
                max_size = 1024
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                im_scale = float(target_size) / float(im_size_min)
                # prevent bigger axis from being more than max_size:
                if np.round(im_scale * im_size_max) > max_size:
                    im_scale = float(max_size) / float(im_size_max)
                scales = [im_scale]
            else:
                do_flip = True
                TEST_SCALES = [500, 800, 1100, 1400, 1700]
                target_size = 800
                max_size = 1200
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                im_scale = float(target_size) / float(im_size_min)
                # prevent bigger axis from being more than max_size:
                if np.round(im_scale * im_size_max) > max_size:
                    im_scale = float(max_size) / float(im_size_max)
                scales = [float(scale) / target_size * im_scale for scale in TEST_SCALES]

            faces, landmarks = self.detector.detect(img, self.threshold, scales=scales, do_flip=do_flip)

            det_boxes = faces[np.newaxis, :, :4]
            det_scores = faces[np.newaxis, :, 4]
            det_labels = [np.zeros_like(det_scores, dtype=np.int)]
            gt_boxes = labels[np.newaxis, :, :4]
            gt_labels = labels[np.newaxis, :, 4]
            gt_difficult = np.zeros_like(gt_labels)
            eval_metric.update(det_boxes, det_labels, det_scores, gt_boxes, gt_labels, gt_difficult)
        map_name, mean_ap = eval_metric.get()

        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        if verbose:
            print(val_msg)
        eval_dict = {k.lower(): v for k, v in zip(map_name, mean_ap)}
        return eval_dict

    def __prepare_dataset(self, dataset):
        """
        Prepares the WIDER Face dataset for training. Additional training annotations are downloaded if they don't already
        exist in the dataset folder.
        :param dataset: object containing WiderFaceDataset
        :type dataset: opendr.perception.object_detection_2d.datasets.WiderFaceDataset
        :return: the dataset is returned as-is, apart from a minor modification in the dataset_type attribute,
        made for consistency reasons with the original implementation
        :rtype:
        """
        if issubclass(type(dataset), WiderFaceDataset):
            dataset.dataset_type = ''.join(dataset.dataset_type.split('_'))
            if not os.path.exists(os.path.join(dataset.root, "WIDER_train", "labels.txt")):
                print("Landmark annotations not found, downloading to dataset root...")
                self.download(dataset.root, mode="annotations")
            return dataset
        else:
            return ValueError("Only WIDER face dataset is supported for this detector")

    @staticmethod
    def __prepare_val_dataset(dataset):
        """
        Prepares any face DetectionDataset for evaluation.
        :param dataset: evaluation dataset object
        :type dataset: opendr.perception.obejct_detection_2d.datasets.DetectionDataset
        :return: returns the converted dataset and recall metric
        :rtype: opendr.perception.obejct_detection_2d.datasets.DetectionDataset,
        opendr.perception.object_detection_2d.retinaface.algorithm.eval_recall.FaceDetectionRecallMetric
        """
        if issubclass(type(dataset), WiderFaceDataset):
            dataset.dataset_type = ''.join(dataset.dataset_type.split('_'))
            eval_metric = FaceDetectionRecallMetric()
            return dataset, eval_metric
        elif issubclass(type(dataset), DetectionDataset):
            eval_metric = FaceDetectionRecallMetric()
            return dataset, eval_metric
        else:
            return ValueError("Dataset must be subclass of the DetectionDataset base class.")

    @staticmethod
    def __get_fixed_params(symbol):
        """
        Makes necessary modifications to the network's symbolic structure before training
        """
        if not config.LAYER_FIX:
            return []
        fixed_param_names = []
        idx = 0
        for name in symbol.list_arguments():
            if idx < 7 and name != 'data':
                fixed_param_names.append(name)
            if name.find('upsampling') >= 0:
                fixed_param_names.append(name)
            idx += 1
        return fixed_param_names

    def infer(self, img, threshold=0.8, nms_threshold=0.4, scales=[1024, 1980], mask_thresh=0.8):
        """
        Performs inference on a single image and returns the resulting bounding boxes.
        :param img: image to perform inference on
        :type img: opendr.engine.data.Image
        :param threshold: confidence threshold
        :type threshold: float, optional
        :param nms_threshold: NMS threshold
        :type nms_threshold: float, optional
        :param scales: inference scales
        :type scales: list, optional
        :param mask_thresh: mask confidence threshold, only use when backbone is 'mnet'
        :type mask_thresh: float, optional
        :return: list of bounding boxes
        :rtype: BoundingBoxList
        """
        if self.detector is None:
            assert "Detector must be loaded with load() before inference."

        self.detector.nms_threshold = nms_threshold

        if not isinstance(img, Image):
            img = Image(img)
        _img = img.convert("channels_last", "rgb")

        im_shape = _img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        flip = False

        faces, landmarks = self.detector.detect(_img, threshold, scales=scales, do_flip=flip)
        faces = np.hstack([faces, np.zeros((faces.shape[0], 1))])
        bboxes = BoundingBoxList([])
        for face in faces:
            if face.shape[0] > 4:
                mask = int(face[5] > mask_thresh)
            # faces in xywhc format, convert to BoundingBoxs
            bbox = BoundingBox(left=face[0], top=face[1],
                               width=face[2] - face[0],
                               height=face[3] - face[1],
                               name=mask, score=face[4])

            bboxes.data.append(bbox)

        # return faces, landmarks
        return bboxes

    def save(self, path, verbose=False):
        """
        This method is used to save a trained model.
        Provided with the path, absolute or relative, including a *folder* name, it creates a directory with the name
        of the *folder* provided and saves the model inside with a proper format and a .json file with metadata.

        :param path: for the model to be log, including the folder name
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        if self._model is None:
            raise UserWarning("No model is loaded, cannot save.")

        model_name = os.path.basename(path)
        os.makedirs(path, exist_ok=True)

        model_metadata = {"model_paths": [], "framework": "mxnet", "format": "", "has_data": False,
                          "inference_params": {}, "optimized": None, "optimizer_info": {},
                          }
        _sym, arg, aux = self.__prepare_detector(self._model)
        model_path = os.path.join(path, model_name)
        mx.model.save_checkpoint(model_path, 0, _sym, arg, aux)
        if verbose:
            print("Saved model.")

        model_metadata["model_paths"] = ['%s-symbol.json' % model_name,
                                         '%s-%04d.params' % (model_name, 0)]
        model_metadata["optimized"] = False
        model_metadata["format"] = "checkpoint"

        with open(os.path.join(path, model_name + ".json"), 'w') as outfile:
            json.dump(model_metadata, outfile)
        if verbose:
            print("Saved model metadata.")

    def load(self, path, verbose=True):
        """
        Loads the model from inside the path provided, based on the metadata .json file included.

        :param path: path of the directory the model was log
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        # first, get model_name from path
        model_name = os.path.basename(path)

        with open(os.path.join(path, model_name + ".json")) as metadata_file:
            metadata = json.load(metadata_file)
        model_name = metadata['model_paths'][0].split('-')[0]
        model_path = os.path.join(path, model_name)
        if verbose:
            print("Loading model from path: ", model_path)

        generate_config(self.backbone, 'retinaface')
        self.detector = RetinaFace(prefix=model_path, ctx_id=self.gpu_id, network=self.net)
        self._model = self.detector.model

        if verbose:
            print("Loaded mxnet model.")

    def download(self, path=None, mode="pretrained", verbose=True,
                 url=OPENDR_SERVER_URL + "perception/object_detection_2d/retinaface/"):
        """
        Downloads all files necessary for inference, evaluation and training. Valid mode options: ["pretrained", "images",
        "backbone", "annotations"]
        :param path: folder to which files will be downloaded, if None self.temp_path will be used
        :type path: str, optional
        :param mode: one of: ["pretrained", "images", "backbone", "annotations"], where "pretrained" downloads a pretrained
        network depending on the self.backbone type, "images" downloads example inference data, "backbone" downloads a
        pretrained resnet backbone for training, and "annotations" downloads additional annotation files for training
        :type mode: str, optional
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional
        :param url: URL to file location on FTP server
        :type url: str, optional
        """
        valid_modes = ["pretrained", "images", "backbone", "annotations", "test_data"]
        if mode not in valid_modes:
            raise UserWarning("Parameter 'mode' should be one of:", valid_modes)

        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":
            model_name = "retinaface_{}".format(self.backbone)
            if verbose:
                print("Downloading pretrained files for {}".format(model_name))
            path = os.path.join(path, model_name)
            if not os.path.exists(path):
                os.makedirs(path)

            if verbose:
                print("Downloading pretrained model...")

            file_url = os.path.join(url, "pretrained", model_name, "{}.json".format(model_name))
            if verbose:
                print("Downloading metadata...")
            file_path = os.path.join(path, "{}.json".format(model_name))
            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

            if verbose:
                print("Downloading params...")
            file_url = os.path.join(url, "pretrained", model_name, "{}-0000.params".format(model_name))
            file_path = os.path.join(path, "{}-0000.params".format(model_name))
            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

            if verbose:
                print("Downloading symbol...")
            file_url = os.path.join(url, "pretrained", model_name, "{}-symbol.json".format(model_name))
            file_path = os.path.join(path, "{}-symbol.json".format(model_name))
            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

        elif mode == "images":
            file_url = os.path.join(url, "images", "cov4.jpg")
            if verbose:
                print("Downloading example image...")
            file_path = os.path.join(path, "cov4.jpg")
            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

        elif mode == "annotations":
            if verbose:
                print("Downloading training annotations...")
            for subset in ["train", "val", "test"]:
                file_url = os.path.join(url, "annotations", "WIDER_{}".format(subset), "label.txt")
                file_path = os.path.join(path, "WIDER_{}".format(subset), "label.txt")
                if not os.path.exists(file_path):
                    urlretrieve(file_url, file_path)

        elif mode == "backbone":
            if verbose:
                print("Downloading resnet backbone...")
            file_url = os.path.join(url, "backbone", "resnet-50-symbol.json")
            file_path = os.path.join(path, "resnet-50-symbol.json")
            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

            file_url = os.path.join(url, "backbone", "resnet-50-0000.params")
            file_path = os.path.join(path, "resnet-50-0000.params")
            if not os.path.exists(file_path):
                urlretrieve(file_url, file_path)

        elif mode == "test_data":
            if verbose:
                print("Downloading data for unit tests...")
            # fake label.txt files
            for subset in ["train", "val", "test"]:
                file_url = os.path.join(url, "test_data", "WIDER_{}".format(subset), "label.txt")
                if verbose:
                    print(file_url)
                file_path = os.path.join(path, "WIDER_{}".format(subset), "label.txt")
                if not os.path.exists(file_path):
                    os.makedirs(os.path.join(path, "WIDER_{}".format(subset)), exist_ok=True)
                    urlretrieve(file_url, file_path)
            # single training image
            file_url = os.path.join(url, "test_data", "WIDER_train", "images",
                                    "0--Parade", "0_Parade_marchingband_1_849.jpg")
            if verbose:
                print(file_url)
            file_path = os.path.join(path, "WIDER_train", "images",
                                     "0--Parade", "0_Parade_marchingband_1_849.jpg")
            if not os.path.exists(file_path):
                print("t")
                os.makedirs(os.path.join(path, "WIDER_train", "images", "0--Parade"), exist_ok=True)
                print("t")
                urlretrieve(file_url, file_path)
                if verbose:
                    print("Downloaded")
            # annotations
            file_url = os.path.join(url, "test_data", "wider_face_split",
                                    "wider_face_train_bbx_gt.txt")
            if verbose:
                print(file_url)
            file_path = os.path.join(path, "wider_face_split",
                                     "wider_face_train_bbx_gt.txt")
            if not os.path.exists(file_path):
                os.makedirs(os.path.join(path, "wider_face_split"), exist_ok=True)
                urlretrieve(file_url, file_path)

    def optimize(self, target_device):
        """This method is not used in this implementation."""
        return NotImplementedError

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError
