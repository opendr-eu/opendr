# Copyright 2020-2022 OpenDR European Project
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
#
# general imports
import os
import time
import json
import numpy as np
from multiprocessing import Pool
import cv2
from tqdm import tqdm
from urllib.request import urlretrieve

# gluoncv imports
import mxnet as mx
from mxnet import gluon, nd, autograd
from gluoncv import utils as gutils
from gluoncv import model_zoo
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import SiamRPNTracker as build_tracker
from gluoncv.loss import SiamRPNLoss
from gluoncv.utils import LRScheduler, LRSequential, split_and_load
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import get_axis_aligned_bbox
from gluoncv.data.tracking_data.track import TrkDataset
from gluoncv.utils.metrics.tracking import OPEBenchmark
from gluoncv.data.otb.tracking import OTBTracking

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import TrackingAnnotation
from opendr.engine.datasets import ExternalDataset
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.datasets import DatasetIterator

gutils.random.seed(0)
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


class SiamRPNLearner(Learner):
    def __init__(self, device='cuda', n_epochs=50, num_workers=1, warmup_epochs=2,
                 lr=1e-3, weight_decay=0, momentum=0.9, cls_weight=1., loc_weight=1.2,
                 batch_size=32, temp_path=''):
        """
        SiamRPN Tracker Learner
        :param device: Either 'cpu' or 'cuda'. If a specific GPU is to be used, can be of the form 'cuda:#'
        :type device: str, optional
        :param n_epochs: Total number of epochs to train for
        :type n_epochs: int, optional
        :param num_workers: Number of threads used to load the train dataset or perform evaluation
        :type num_workers: int, optional
        :param warmup_epochs: Number of epochs during which the learning rate is annealer to `lr`
        :type warmup_epochs: int, optional
        :param lr: Initial learning rate, after warmup_epochs
        :type lr: float, optional
        :param weight_decay: Weight decay factor
        :type weight_decay: float, optional
        :param momentum: Optimizer momentum
        :type momentum: float, optional
        :param cls_weight: Weighs the classification loss
        :type cls_weight: float, optional
        :param loc_weight: Weights the localization loss
        :type loc_weight: float, optional
        :param batch_size: Batch size for training
        :type batch_size: int, optional
        :param temp_path: path to where relevant data and weights are downloaded
        :type temp_path: str, optional
        """
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight
        self.warmup_epochs = warmup_epochs
        self.num_workers = num_workers
        self.n_epochs = n_epochs
        backbone = 'siamrpn_alexnet_v2_otb15'
        super(SiamRPNLearner, self).__init__(device=device, backbone=backbone, lr=lr,
                                             batch_size=batch_size, temp_path=temp_path)
        self.weight_decay = weight_decay
        self.momentum = momentum

        if 'cuda' in self.device:
            if mx.context.num_gpus() > 0:
                if self.device == 'cuda':
                    self.ctx = mx.gpu(0)
                else:
                    self.ctx = mx.gpu(int(self.device.split(':')[1]))
            else:
                self.ctx = mx.cpu()
        else:
            self.ctx = mx.cpu()

        self.__create_model()
        self.tracker = build_tracker(self._model)

    def __create_model(self, pretrained=True):
        """base model creation"""
        self._model = model_zoo.get_model(self.backbone, ctx=self.ctx, pretrained=pretrained)

    def fit(self, dataset, log_interval=20, n_gpus=1,
            val_dataset=None, logging_path='', silent=True, verbose=True):
        """
        Train the tracker on a new dataset.
        :param dataset: training dataset
        :type dataset: ExternalDataset or supported DatasetIterator
        :param log_interval: Train loss is printed after log_interval iterations
        :type log_interval: int
        :param n_gpus: Number of GPUs to train with if device is set to GPU
        :type n_gpus: int
        :param verbose: if set to True, additional information is printed to STDOUT, defaults to True
        :type verbose: bool
        :param val_dataset: ignored
        :type val_dataset: ExternalDataset, optional
        :param logging_path: ignored
        :type logging_path: str, optional
        :param silent: ignored
        :type silent: str, optional
        :param verbose: if set to True, additional information is printed to STDOUT, defaults to True
        :type verbose: bool
        :return: returns stats regarding the training process
        :rtype: dict
        """
        dataset = self.__prepare_training_dataset(dataset)
        train_loader = gluon.data.DataLoader(dataset,
                                             batch_size=self.batch_size,
                                             last_batch='discard',
                                             num_workers=self.num_workers)

        if self.device.startswith('cuda'):
            if ':' in self.device:
                _, gpu_no = self.device.split(':')
                ctx = [mx.gpu(int(gpu_no))]
            else:
                ctx = [mx.gpu(i) for i in range(n_gpus)]
        else:
            ctx = [mx.cpu(0)]

        self._model = model_zoo.get_model(self.backbone, bz=self.batch_size, is_train=True, ctx=ctx,
                                          pretrained=True)

        criterion = SiamRPNLoss(self.batch_size)
        step_epoch = [10 * i for i in range(0, self.n_epochs, 10)]
        num_batches = len(train_loader)
        lr_scheduler = LRSequential([LRScheduler(mode='step',
                                                 base_lr=0.005,
                                                 target_lr=0.01,
                                                 nepochs=self.warmup_epochs,
                                                 iters_per_epoch=num_batches,
                                                 step_epoch=step_epoch,
                                                 ),
                                     LRScheduler(mode='poly',
                                                 base_lr=0.01,
                                                 target_lr=0.005,
                                                 nepochs=self.n_epochs - self.warmup_epochs,
                                                 iters_per_epoch=num_batches,
                                                 step_epoch=[e - self.warmup_epochs for e in step_epoch],
                                                 power=0.02)])

        optimizer_params = {'lr_scheduler': lr_scheduler,
                            'wd': self.weight_decay,
                            'momentum': self.momentum,
                            'learning_rate': self.lr}
        optimizer = gluon.Trainer(self._model.collect_params(), 'sgd', optimizer_params)
        train_dict = {
            'loss_total': [],
            'loss_loc': [],
            'loss_cls': []
        }

        for epoch in range(self.n_epochs):
            loss_total_val = 0
            loss_loc_val = 0
            loss_cls_val = 0
            batch_time = time.time()
            for i, data in enumerate(train_loader):
                template, search, label_cls, label_loc, label_loc_weight = self.__train_batch_fn(data, ctx)
                cls_losses = []
                loc_losses = []
                total_losses = []

                with autograd.record():
                    for j in range(len(ctx)):
                        cls, loc = self._model(template[j], search[j])
                        label_cls_temp = label_cls[j].reshape(-1).asnumpy()
                        pos_index = np.argwhere(label_cls_temp == 1).reshape(-1)
                        neg_index = np.argwhere(label_cls_temp == 0).reshape(-1)
                        if len(pos_index):
                            pos_index = nd.array(pos_index, ctx=ctx[j])
                        else:
                            pos_index = nd.array(np.array([]), ctx=ctx[j])
                        if len(neg_index):
                            neg_index = nd.array(neg_index, ctx=ctx[j])
                        else:
                            neg_index = nd.array(np.array([]), ctx=ctx[j])
                        cls_loss, loc_loss = criterion(cls, loc, label_cls[j], pos_index, neg_index,
                                                       label_loc[j], label_loc_weight[j])
                        total_loss = self.cls_weight * cls_loss + self.loc_weight * loc_loss
                        cls_losses.append(cls_loss)
                        loc_losses.append(loc_loss)
                        total_losses.append(total_loss)

                    mx.nd.waitall()
                    autograd.backward(total_losses)
                optimizer.step(self.batch_size)
                loss_total_val += sum([l.mean().asscalar() for l in total_losses]) / len(total_losses)
                loss_loc_val += sum([l.mean().asscalar() for l in loc_losses]) / len(loc_losses)
                loss_cls_val += sum([l.mean().asscalar() for l in cls_losses]) / len(cls_losses)
                if i % log_interval == 0 and verbose:
                    print('Epoch %d iteration %04d/%04d: loc loss %.3f, cls loss %.3f, \
                                     training loss %.3f, batch time %.3f' %
                          (epoch, i, len(train_loader), loss_loc_val / (i + 1), loss_cls_val / (i + 1),
                           loss_total_val / (i + 1), time.time() - batch_time))
                    batch_time = time.time()
                mx.nd.waitall()
            train_dict['loss_total'].append(loss_total_val)
            train_dict['loss_loc'].append(loss_loc_val)
            train_dict['loss_cls'].append(loss_cls_val)
        return train_dict

    def eval(self, dataset):
        """
        Evaluate the current model on the OTB dataset. Measures success and FPS.
        :param dataset: Dataset for evaluation.
        :type dataset: `ExternalDataset`
        :return: returns stats regarding evaluation
        :rtype: dict
        """
        tracker_name = self.backbone

        dataset = self.__prepare_validation_dataset(dataset)

        self._model.collect_params().reset_ctx(self.ctx)
        self.tracker = build_tracker(self._model)

        # iterate through dataset
        fps = np.zeros(len(dataset))
        for v_idx, video in enumerate(dataset):
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    x_max, y_max, gt_w, gt_t = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [x_max - (gt_w - 1) / 2, y_max - (gt_t - 1) / 2, gt_w, gt_t]
                    self.tracker.init(img, gt_bbox_, self.ctx)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    pred_bboxes.append(pred_bbox)
                else:
                    outputs = self.tracker.track(img, self.ctx)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            toc /= cv2.getTickFrequency()

            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format
                  (v_idx + 1, video.name, toc, len(video) / toc))
            video.pred_trajs[tracker_name] = pred_bboxes
            fps[v_idx] = len(video) / toc

        mean_fps = np.mean(fps)
        benchmark = OPEBenchmark(dataset)
        trackers = [tracker_name]
        success_ret = {}
        with Pool(processes=self.num_workers) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success, trackers),
                            desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=self.num_workers) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision, trackers),
                            desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)

        tracker_auc = {}
        for tracker_name in success_ret.keys():
            auc = np.mean(list(success_ret[tracker_name].values()))
            tracker_auc[tracker_name] = auc

        tracker_name_len = max((max([len(x) for x in success_ret.keys()]) + 2), 12)
        header = ("|{:^" + str(tracker_name_len) + "}|{:^9}|{:^9}|").format("Tracker name", "Success", "FPS")
        formatter = "|{:^" + str(tracker_name_len) + "}|{:^9.3f}|{:^9.1f}|"
        print('-' * len(header))
        print(header)
        print('-' * len(header))
        success = tracker_auc[tracker_name]
        print(formatter.format(tracker_name, success, mean_fps))
        print('-' * len(header))
        eval_dict = {
            'success': success,
            'fps': mean_fps
        }
        return eval_dict

    def infer(self, img, init_box=None):
        """
        Performs inference on an input image and returns the resulting bounding box.
        :param img: image to perform inference on
        :type img: opendr.engine.data.Image
        :param init_box: If provided, it is used to initialized the tracker on the contained object
        :type init_box: TrackingAnnotation
        :return: list of bounding boxes
        :rtype: BoundingBoxList
        """
        if isinstance(img, Image):
            img = img.opencv()

        if isinstance(init_box, TrackingAnnotation) and init_box is not None:
            # initialize tracker
            gt_bbox_ = [init_box.left, init_box.top, init_box.width, init_box.height]
            self.tracker.init(img, gt_bbox_, ctx=self.ctx)
            pred_bbox = gt_bbox_
        else:
            outputs = self.tracker.track(img, ctx=self.ctx)
            pred_bbox = outputs['bbox']

        pred_bbox = list(map(int, pred_bbox))
        return TrackingAnnotation(left=pred_bbox[0], top=pred_bbox[1],
                                  width=pred_bbox[2], height=pred_bbox[3], name=0, id=0)

    def save(self, path, verbose=False):
        """
        Method for saving the current model in the path provided.
        :param path: path to folder where model will be saved
        :type path: str
        :param verbose: whether to print a success message or not, defaults to False
        :type verbose: bool, optional
        """
        os.makedirs(path, exist_ok=True)

        model_name = os.path.basename(path)
        if verbose:
            print(model_name)
        metadata = {"model_paths": [], "framework": "mxnet", "format": "params",
                    "has_data": False, "inference_params": {}, "optimized": False,
                    "optimizer_info": {}, "backbone": self.backbone}
        param_filepath = model_name + ".params"
        metadata["model_paths"].append(param_filepath)

        self._model.save_parameters(os.path.join(path, metadata["model_paths"][0]))
        if verbose:
            print("Model parameters saved.")

        with open(os.path.join(path, model_name + '.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        if verbose:
            print("Model metadata saved.")
        return True

    def load(self, path, verbose=False):
        """
        Loads the model from the path provided, based on the metadata .json file included.
        :param path: path of the directory where the model was saved
        :type path: str
        :param verbose: whether to print a success message or not, defaults to False
        :type verbose: bool, optional
        """
        model_name = os.path.basename(os.path.normpath(path))
        if verbose:
            print("Model name:", model_name, "-->", os.path.join(path, model_name + ".json"))
        with open(os.path.join(path, model_name + ".json")) as f:
            metadata = json.load(f)

        self.backbone = metadata["backbone"]
        self.__create_model(pretrained=False)

        self._model.load_parameters(os.path.join(path, metadata["model_paths"][0]))
        self._model.collect_params().reset_ctx(self.ctx)
        self._model.hybridize(static_alloc=True, static_shape=True)
        if verbose:
            print("Loaded parameters and metadata.")
        return True

    def download(self, path=None, mode="pretrained", verbose=False,
                 url=OPENDR_SERVER_URL + "/perception/object_tracking_2d/siamrpn/",
                 overwrite=False):
        """
        Downloads all files necessary for inference, evaluation and training. Valid mode options are: ["pretrained",
        "video", "test_data", "otb2015"].
        :param path: folder to which files will be downloaded, if None self.temp_path will be used
        :type path: str, optional
        :param mode: one of: ["pretrained", "video", "test_data", "otb2015"], where "pretrained" downloads a pretrained
        network, "video" downloads example inference data, "test_data" downloads a very small train/eval subset, and
         "otb2015" downloads the OTB dataset
        :type mode: str, optional
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional
        :param url: URL to file location on FTP server
        :type url: str, optional
        :param overwrite: if True, the downloaded files will be overwritten
        :type overwrite:bool, optional
        """
        valid_modes = ["pretrained", "video", "otb2015", "test_data"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":
            path = os.path.join(path, "siamrpn_opendr")
            if not os.path.exists(path):
                os.makedirs(path)

            if verbose:
                print("Downloading pretrained model...")

            file_url = os.path.join(url, "pretrained",
                                    "siamrpn_opendr",
                                    "siamrpn_opendr.json")
            if verbose:
                print("Downloading metadata...")
            file_path = os.path.join(path, "siamrpn_opendr.json")
            if not os.path.exists(file_path) or overwrite:
                urlretrieve(file_url, file_path)

            if verbose:
                print("Downloading params...")
            file_url = os.path.join(url, "pretrained", "siamrpn_opendr",
                                    "siamrpn_opendr.params")
            file_path = os.path.join(path, "siamrpn_opendr.params")
            if not os.path.exists(file_path) or overwrite:
                urlretrieve(file_url, file_path)

        elif mode == "video":
            file_url = os.path.join(url, "video", "tc_Skiing_ce.mp4")
            if verbose:
                print("Downloading example video...")
            file_path = os.path.join(path, "tc_Skiing_ce.mp4")
            if not os.path.exists(file_path) or overwrite:
                urlretrieve(file_url, file_path)

        elif mode == "test_data":
            os.makedirs(os.path.join(path, "Basketball"), exist_ok=True)
            os.makedirs(os.path.join(path, "Basketball", "img"), exist_ok=True)
            # download annotation
            file_url = os.path.join(url, "test_data", "OTBtest.json")
            if verbose:
                print("Downloading annotation...")
            file_path = os.path.join(path, "OTBtest.json")
            if not os.path.exists(file_path) or overwrite:
                urlretrieve(file_url, file_path)
            # download image
            if verbose:
                print("Downloading 100 images...")
            for i in range(100):
                file_url = os.path.join(url, "test_data", "Basketball", "img", f"{i+1:04d}.jpg")
                file_path = os.path.join(path, "Basketball", "img", f"{i+1:04d}.jpg")
                if not os.path.exists(file_path) or overwrite:
                    urlretrieve(file_url, file_path)

        else:
            # mode == 'otb2015'
            from .data_utils.otb import download_otb
            if verbose:
                print('Attempting to download OTB2015 (100 videos)...')
            download_otb(os.path.join(path, "otb2015"), overwrite=overwrite)
            file_url = os.path.join(url, "otb2015", "OTB2015.json")
            if verbose:
                print("Downloading annotation...")
            file_path = os.path.join(path, "otb2015", "OTB2015.json")
            if not os.path.exists(file_path) or overwrite:
                urlretrieve(file_url, file_path)

    @staticmethod
    def __train_batch_fn(data, ctx):
        """split and load data in GPU"""
        template = split_and_load(data[0], ctx_list=ctx, batch_axis=0)
        search = split_and_load(data[1], ctx_list=ctx, batch_axis=0)
        label_cls = split_and_load(data[2], ctx_list=ctx, batch_axis=0)
        label_loc = split_and_load(data[3], ctx_list=ctx, batch_axis=0)
        label_loc_weight = split_and_load(data[4], ctx_list=ctx, batch_axis=0)
        return template, search, label_cls, label_loc, label_loc_weight

    def __prepare_training_dataset(self, dataset):
        """
        Converts `ExternalDataset` or list of `ExternalDatasets` to appropriate format.
        :param dataset: Training dataset(s)
        :type dataset: `ExternalDataset` or list of `ExternalDatasets`
        """
        if isinstance(dataset, list) or isinstance(dataset, tuple):
            frame_range_map = {
                'vid': 100,
                'Youtube_bb': 3,
                'coco': 1,
                'det': 1,
            }
            num_use_map = {
                'vid': 100000,
                'Youtube_bb': -1,
                'coco': -1,
                'det': -1,
            }

            dataset_paths = []
            dataset_names = []
            dataset_roots = []
            dataset_annos = []
            frame_ranges = []
            num_uses = []

            for _dataset in dataset:
                # check if all are ExternalDataset
                if not isinstance(dataset, ExternalDataset):
                    raise TypeError("Only `ExternalDataset` types are supported.")
                # get params
                dataset_paths.append(_dataset.path)
                dataset_names.append(_dataset.dataset_type)
                dataset_roots.append(os.path.join(_dataset.dataset_type, 'crop511'))
                dataset_annos.append(os.path.join(_dataset.dataset_type,
                                                  f'train{"2017" if _dataset.dataset_type == "coco" else ""}.json'))
                frame_ranges.append(frame_range_map[_dataset.dataset_type])
                num_uses.append(num_use_map[_dataset.dataset_type])
            dataset = TrkDataset(dataset[0].path,
                                 dataset_names=dataset_names, detaset_root=dataset_roots,
                                 detaset_anno=dataset_annos, train_epoch=self.n_epochs,
                                 dataset_frame_range=frame_ranges, dataset_num_use=num_uses)
            return dataset

        if isinstance(dataset, ExternalDataset):
            dataset_types = ['vid', 'Youtube_bb', 'coco', 'det']
            assert dataset.dataset_type in dataset_types, f"Unrecognized dataset_type," \
                                                          f" acceptable values: {dataset_types}"
            dataset = TrkDataset(data_path=dataset.path,
                                 dataset_names=[dataset.dataset_type], detaset_root=[f'{dataset.dataset_type}/crop511'],
                                 detaset_anno=[f'{dataset.dataset_type}/'
                                               f'train{"2017" if dataset.dataset_type == "coco" else ""}.json'],
                                 train_epoch=self.n_epochs)
            return dataset

        if issubclass(type(dataset), DatasetIterator):
            return dataset

        if not isinstance(dataset, ExternalDataset):
            raise TypeError("Only `ExternalDataset` and modified `DatasetIterator` types are supported.")

    @staticmethod
    def __prepare_validation_dataset(dataset):
        """
        :param dataset: `ExternalDataset` object containing OTB2015 dataset root and type ('OTB2015')
        :type dataset: ExternalDataset
        """
        if not isinstance(dataset, ExternalDataset):
            raise TypeError("Only `ExternalDataset` types are supported.")
        dataset_types = ["OTB2015", "OTBtest"]
        assert dataset.dataset_type in dataset_types, "Unrecognized dataset type, only OTB2015 is supported currently"
        dataset = OTBTracking(dataset.dataset_type, dataset_root=dataset.path, load_img=False)
        return dataset

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def optimize(self, target_device):
        """This method is not used in this implementation."""
        return NotImplementedError
