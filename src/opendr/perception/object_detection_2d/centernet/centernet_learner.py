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

# General imports
import os
import time
import json
import numpy as np
import warnings
from tqdm import tqdm
from urllib.request import urlretrieve

# gluoncv imports
from gluoncv.data.transforms import presets
from gluoncv.data.batchify import Tuple, Stack, Pad
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
# gcv.utils.check_version('0.6.0')
from gluoncv import model_zoo
from gluoncv import utils as gutils
from gluoncv.utils import LRScheduler, LRSequential
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.datasets import ExternalDataset
from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.engine.constants import OPENDR_SERVER_URL

# algorithm imports
from opendr.perception.object_detection_2d.utils.eval_utils import DetectionDatasetCOCOEval
from opendr.perception.object_detection_2d.datasets.transforms import ImageToNDArrayTransform, transform_test, \
    BoundingBoxListToNumpyArray
from opendr.perception.object_detection_2d.datasets import DetectionDataset

gutils.random.seed(0)


class CenterNetDetectorLearner(Learner):
    supported_backbones = ["resnet50_v1b"]

    def __init__(self, lr=1e-3, epochs=120, batch_size=8, device='cuda', backbone='resnet50_v1b', img_size=512,
                 lr_schedule='step', temp_path='', checkpoint_after_iter=0, checkpoint_load_iter=0,
                 val_after=5, log_after=100, num_workers=8, weight_decay=5e-4, momentum=0.9,
                 scale=1., wh_weight=0.1, center_reg_weight=1.0,
                 lr_decay_epoch='80,100', lr_decay=0.1, warmup_epochs=0, flip_validation=False):
        super(CenterNetDetectorLearner, self).__init__(lr=lr, batch_size=batch_size, lr_schedule=lr_schedule,
                                                       checkpoint_after_iter=checkpoint_after_iter,
                                                       checkpoint_load_iter=checkpoint_load_iter,
                                                       temp_path=temp_path, device=device, backbone=backbone)
        assert self.lr_schedule in ['step', 'cosine', 'poly']
        self.epochs = epochs
        self.log_after = log_after
        self.val_after = val_after
        self.num_workers = num_workers
        self.backbone = backbone.lower()
        self.parent_dir = temp_path
        self.checkpoint_str_format = "checkpoint_epoch_{}.params"

        self.scale = scale
        self.flip_validation = flip_validation
        self.data_shape = img_size

        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.warmup_epochs = warmup_epochs
        self.wh_weight = wh_weight
        self.center_reg_weight = center_reg_weight

        if self.device == 'cuda':
            if mx.context.num_gpus() > 0:
                self.ctx = mx.gpu(0)
            else:
                self.ctx = mx.cpu()
        else:
            self.ctx = mx.cpu()

        self.img_size = img_size
        self.weight_decay = weight_decay
        self.momentum = momentum

        net = model_zoo.get_model('center_net_{}_voc'.format(self.backbone),
                                  pretrained=False,
                                  pretrained_base=True)
        self._model = net
        self.classes = ['None']

    def __create_model(self, classes):
        """
        Base method for detector creation, based on gluoncv implementation.
        :param classes: list of classes contained in the training set
        :type classes: list
        """

        if self._model is None or classes != self.classes:
            self._model = model_zoo.get_model('center_net_{}_voc'.format(self.backbone),
                                              pretrained=False,
                                              pretrained_base=True)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self._model.initialize()
            self._model.collect_params().reset_ctx(self.ctx)
            self._model.hybridize()
        self._model.reset_class(classes)
        self.classes = classes

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        """
        This method is used to train the detector on the WIDER Face dataset. Validation if performed if a val_dataset is
        provided.
        :param dataset: training dataset; custom DetectionDataset types are supported as-is. COCO and Pascal VOC are
        supported as ExternalDataset types, with 'coco' or 'voc' dataset_type attributes.
        :type dataset: DetectionDataset or ExternalDataset
        :param val_dataset: validation dataset object
        :type val_dataset: ExternalDataset or DetectionDataset
        :param logging_path: ignored
        :type logging_path: str, optional
        :param silent: ignored
        :type silent: str, optional
        :param verbose: if set to True, additional information is printed to STDOUT, defaults to True
        :type verbose: bool
        :return: returns stats regarding the training and validation process
        :rtype: dict
        """
        # set save dir for checkpoint saving
        save_prefix = 'centernet_{}_{}'.format(self.backbone, dataset.dataset_type)

        # get dataset in compatible format
        dataset = self.__prepare_dataset(dataset)
        self.__create_model(dataset.classes)
        if verbose:
            print('Saving models as {}'.format(save_prefix))

        # get net & set device
        if self.device == 'cuda':
            if mx.context.num_gpus() > 0:
                ctx = [mx.gpu(0)]
            else:
                ctx = [mx.cpu()]
        else:
            ctx = [mx.cpu()]
        if verbose:
            print("Using device: ", self.device, ctx)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self._model.initialize()

        checkpoints_folder = os.path.join(self.parent_dir, '{}_checkpoints'.format(save_prefix))
        if self.checkpoint_after_iter != 0 and not os.path.exists(checkpoints_folder):
            # User set checkpoint_after_iter so checkpoints need to be created
            # Checkpoints folder was just created
            os.makedirs(checkpoints_folder, exist_ok=True)

        start_epoch = 0
        if self.checkpoint_load_iter > 0:
            # User set checkpoint_load_iter, so they want to load a checkpoint
            # Try to find the checkpoint_load_iter checkpoint
            if verbose:
                print("Resuming training from epoch {}".format(self.checkpoint_load_iter))
            checkpoint_name = self.checkpoint_str_format.format(self.checkpoint_load_iter)
            checkpoint_path = os.path.join(checkpoints_folder, checkpoint_name)
            try:
                self._model.load_params(checkpoint_path)
                start_epoch = self.checkpoint_load_iter
            except FileNotFoundError as e:
                e.strerror = "File " + checkpoint_name + " not found inside checkpoints_folder, " \
                                                         "provided checkpoint_load_iter (" + \
                             str(self.checkpoint_load_iter) + \
                             ") doesn't correspond to a log checkpoint.\nNo such file or directory."
                raise e

        self._model.collect_params().reset_ctx(ctx)
        # lr decay policy
        lr_decay = float(self.lr_decay)
        lr_steps = sorted([int(ls) for ls in self.lr_decay_epoch.split(',') if ls.strip()])
        lr_decay_epoch = [e - self.warmup_epochs for e in lr_steps]
        num_batches = len(dataset) // self.batch_size
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self.lr,
                        nepochs=self.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(self.lr_schedule, base_lr=self.lr,
                        nepochs=self.epochs - self.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=lr_decay, power=2),
        ])

        for k, v in self._model.collect_params('.*bias').items():
            v.wd_mult = 0.0
        trainer = gluon.Trainer(
            self._model.collect_params(), 'adam',
            {'wd': self.weight_decay,
             'lr_scheduler': lr_scheduler})

        width, height = self.data_shape, self.data_shape
        num_class = len(dataset.classes)
        batchify_fn = Tuple([Stack() for _ in range(6)])  # stack image, cls_targets, box_targets
        train_loader = gluon.data.DataLoader(
            dataset.transform(presets.center_net.CenterNetDefaultTrainTransform(
                width, height, num_class=num_class, scale_factor=self._model.scale)),
            self.batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=self.num_workers)

        heatmap_loss = gcv.loss.HeatmapFocalLoss(from_logits=True)
        wh_loss = gcv.loss.MaskedL1Loss(weight=self.wh_weight)
        center_reg_loss = gcv.loss.MaskedL1Loss(weight=self.center_reg_weight)
        heatmap_loss_metric = mx.metric.Loss('HeatmapFocal')
        wh_metric = mx.metric.Loss('WHL1')
        center_reg_metric = mx.metric.Loss('CenterRegL1')

        training_dict = {"HeatmapFocal": [], "WHL1": [], "CenterRegL1": [], "val_map": []}

        for epoch in range(start_epoch, self.epochs):
            print('[Epoch {}/{} lr={}]'.format(epoch, self.epochs, trainer.learning_rate))

            wh_metric.reset()
            center_reg_metric.reset()
            tic = time.time()
            self._model.hybridize()

            for i, batch in enumerate(train_loader):
                split_data = [gluon.utils.split_and_load(batch[ind], ctx_list=ctx, batch_axis=0) for ind in range(6)]
                # data, heatmap_targets, wh_targets, wh_masks, center_reg_targets, center_reg_masks = split_data
                # batch_size = self.batch_size
                with autograd.record():
                    sum_losses = []
                    heatmap_losses = []
                    wh_losses = []
                    center_reg_losses = []
                    wh_preds = []
                    center_reg_preds = []
                    for x, heatmap_target, wh_target, wh_mask, center_reg_target, center_reg_mask in zip(*split_data):
                        heatmap_pred, wh_pred, center_reg_pred = self._model(x)
                        wh_preds.append(wh_pred)
                        center_reg_preds.append(center_reg_pred)
                        wh_losses.append(wh_loss(wh_pred, wh_target, wh_mask))
                        center_reg_losses.append(center_reg_loss(center_reg_pred, center_reg_target, center_reg_mask))
                        heatmap_losses.append(heatmap_loss(heatmap_pred, heatmap_target))
                        curr_loss = heatmap_losses[-1] + wh_losses[-1] + center_reg_losses[-1]
                        sum_losses.append(curr_loss)
                    autograd.backward(sum_losses)
                trainer.step(len(sum_losses))  # step with # gpus

                heatmap_loss_metric.update(0, heatmap_losses)
                wh_metric.update(0, wh_losses)
                center_reg_metric.update(0, center_reg_losses)
                if i % self.log_after == self.log_after - 1:
                    name2, loss2 = wh_metric.get()
                    name3, loss3 = center_reg_metric.get()
                    name4, loss4 = heatmap_loss_metric.get()
                    training_dict[name2].append(loss2)
                    training_dict[name3].append(loss3)
                    training_dict[name4].append(loss4)
                    print(
                        '[Epoch {}][Batch {}], LR={}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                            epoch, i, trainer.learning_rate, name2, loss2, name3, loss3,
                            name4, loss4))

            name2, loss2 = wh_metric.get()
            name3, loss3 = center_reg_metric.get()
            name4, loss4 = heatmap_loss_metric.get()

            print('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time() - tic), name2, loss2, name3, loss3, name4, loss4))

            if epoch % self.val_after == self.val_after - 1 and val_dataset is not None:
                if verbose:
                    print("Model evaluation at epoch {}".format(epoch))
                eval_dict = self.eval(val_dataset)
                training_dict["val_map"].append(eval_dict["map"])

            if self.checkpoint_after_iter > 0 and epoch % self.checkpoint_after_iter == self.checkpoint_after_iter - 1:
                if verbose:
                    print('Saving model at epoch {}'.format(epoch))
                checkpoint_name = self.checkpoint_str_format.format(epoch)
                checkpoint_filepath = os.path.join(checkpoints_folder, checkpoint_name)
                self._model.save_parameters(checkpoint_filepath)

        return training_dict

    def eval(self, dataset, use_subset=False, subset_size=100, verbose=False):
        """
        This method performs evaluation on a given dataset and returns a dictionary with the evaluation results.
        :param dataset: dataset object, to perform evaluation on
        :type dataset: opendr.perception.object_detection_2d.datasets.DetectionDataset or opendr.engine.data.ExternalDataset
        :return: dictionary containing evaluation metric names nad values
        :param use_subset: if True, only a subset of the dataset is evaluated, defaults to False
        :type use_subset: bool, optional
        :param subset_size: if use_subset is True, subset_size controls the size of the subset to be evaluated
        :type subset_size: int, optional
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional
        :rtype: dict
        """
        autograd.set_training(False)

        # NOTE: multi-gpu is a little bugged
        if self.device == 'cuda':
            if mx.context.num_gpus() > 0:
                ctx = [mx.gpu(0)]
            else:
                ctx = [mx.cpu()]
        else:
            ctx = [mx.cpu()]
        print(self.device, ctx)

        dataset, eval_metric = self.__prepare_val_dataset(dataset, data_shape=self.img_size)

        eval_metric.reset()

        val_transform = presets.center_net.CenterNetDefaultValTransform(self.img_size, self.img_size)
        dataset = dataset.transform(val_transform)

        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        if not use_subset:
            if verbose:
                print('Evaluation on entire dataset...')
            val_loader = gluon.data.DataLoader(
                dataset, self.batch_size, shuffle=False, batchify_fn=val_batchify_fn, last_batch='keep',
                num_workers=self.num_workers)
        else:
            print('Evaluation on subset of dataset...')
            val_loader = gluon.data.DataLoader(
                dataset, self.batch_size, sampler=gluon.data.RandomSampler(subset_size),
                batchify_fn=val_batchify_fn, last_batch='keep',
                num_workers=self.num_workers
            )

        self._model.flip_test = self.flip_validation
        mx.nd.waitall()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self._model.initialize()
        self._model.collect_params().reset_ctx(ctx)
        self._model.hybridize()

        for batch in tqdm(val_loader, total=len(val_loader)):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = self._model(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_labels = y.slice_axis(axis=-1, begin=4, end=5)
                gt_ids.append(gt_labels)
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else np.zeros(gt_labels.shape))

            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        map_name, mean_ap = eval_metric.get()

        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        print(val_msg)
        eval_dict = {k.lower(): v for k, v in zip(map_name, mean_ap)}
        return eval_dict

    def infer(self, img, threshold=0.2, keep_size=True):
        """
        Performs inference on a single image and returns the resulting bounding boxes.
        :param img: image to perform inference on
        :type img: opendr.engine.data.Image
        :param threshold: confidence threshold
        :type threshold: float, optional
        :param keep_size: if True, the image is not resized to fit the data shape used during training
        :type keep_size: bool, optional
        :return: list of bounding boxes
        :rtype: BoundingBoxList
        """

        self._model.set_nms(nms_thresh=0.45, nms_topk=400)

        if not isinstance(img, Image):
            img = Image(img)
        _img = img.convert("channels_last", "rgb")

        height, width, _ = _img.shape
        img_mx = mx.image.image.nd.from_numpy(np.float32(_img))

        if keep_size:
            x, img_mx = transform_test(img_mx)
        else:
            x, img_mx = presets.center_net.transform_test(img_mx, short=self.img_size)

        h_mx, w_mx, _ = img_mx.shape
        x = x.as_in_context(self.ctx)
        class_IDs, scores, boxes = self._model(x)

        class_IDs = class_IDs[0, :].asnumpy()
        scores = scores[0, :].asnumpy()
        mask = np.where((class_IDs >= 0) & (scores > threshold))[0]
        if mask.size == 0:
            return BoundingBoxList([])
        scores = scores[mask, np.newaxis]
        class_IDs = class_IDs[mask, np.newaxis]
        boxes = boxes[0, mask, :].asnumpy()
        boxes[:, [0, 2]] /= w_mx
        boxes[:, [1, 3]] /= h_mx
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height

        bounding_boxes = BoundingBoxList([])
        for idx, box in enumerate(boxes):
            bbox = BoundingBox(left=box[0], top=box[1],
                               width=box[2] - box[0],
                               height=box[3] - box[1],
                               name=class_IDs[idx, :],
                               score=scores[idx, :])
            bounding_boxes.data.append(bbox)
        return bounding_boxes

    def save(self, path, verbose=False):
        """
        Method for saving the current model in the path provided.
        :param path: path to folder where model will be saved
        :type path: str
        :param verbose: whether to print a success message or not, defaults to False
        :type verbose: bool, optional
        """
        if self._model is None:
            raise UserWarning("No model is loaded, cannot save.")

        model_name = os.path.basename(path)
        os.makedirs(path, exist_ok=True)

        model_metadata = {"model_paths": [model_name + ".params"], "framework": "mxnet", "format": "params", "has_data": False,
                          "inference_params": {}, "optimized": False, "optimizer_info": {}, "backbone": self.backbone,
                          "classes": self.classes}

        self._model.save_parameters(os.path.join(path, model_metadata["model_paths"][0]))
        if verbose:
            print("Saved model.")
        with open(os.path.join(path, model_name + ".json"), 'w') as outfile:
            json.dump(model_metadata, outfile)
        if verbose:
            print("Saved model metadata.")

    def load(self, path, verbose=True):
        """
        Loads the model from the path provided, based on the metadata .json file included.
        :param path: path of the directory where the model was saved
        :type path: str
        :param verbose: whether to print a success message or not, defaults to False
        :type verbose: bool, optional
        """
        # first, get model_name from path
        model_name = os.path.basename(path)

        with open(os.path.join(path, model_name + ".json")) as metadata_file:
            metadata = json.load(metadata_file)

        self.backbone = metadata["backbone"]
        if not metadata["optimized"]:
            self.__create_model(metadata["classes"])
            self._model.load_parameters(os.path.join(path, metadata["model_paths"][0]))
            self._model.collect_params().reset_ctx(self.ctx)
            self._model.hybridize()
            if verbose:
                print("Loaded mxnet model.")

    def download(self, path=None, mode="pretrained", verbose=False,
                 url=OPENDR_SERVER_URL + "/perception/object_detection_2d/centernet/"):
        """
        Downloads all files necessary for inference, evaluation and training. Valid mode options are: ["pretrained",
        "images", "test_data"].
        :param path: folder to which files will be downloaded, if None self.temp_path will be used
        :type path: str, optional
        :param mode: one of: ["pretrained", "images", "test_data"], where "pretrained" downloads a pretrained
        network depending on the self.backbone type, "images" downloads example inference data, "backbone" downloads a
        pretrained resnet backbone for training, and "annotations" downloads additional annotation files for training
        :type mode: str, optional
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional
        :param url: URL to file location on FTP server
        :type url: str, optional
        """
        valid_modes = ["pretrained", "images", "test_data"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":
            path = os.path.join(path, "centernet_default")
            if not os.path.exists(path):
                os.makedirs(path)

            if verbose:
                print("Downloading pretrained model...")

            file_url = os.path.join(url, "pretrained",
                                    "centernet_voc",
                                    "centernet_voc.json")
            if verbose:
                print("Downloading metadata...")
            urlretrieve(file_url, os.path.join(path, "centernet_default.json"))

            if verbose:
                print("Downloading params...")
            file_url = os.path.join(url, "pretrained", "centernet_voc",
                                    "centernet_voc.params")

            urlretrieve(file_url,
                        os.path.join(path, "centernet_voc.params"))

        elif mode == "images":
            file_url = os.path.join(url, "images", "bicycles.jpg")
            if verbose:
                print("Downloading example image...")
            urlretrieve(file_url, os.path.join(path, "bicycles.jpg"))

        elif mode == "test_data":
            os.makedirs(os.path.join(path, "test_data"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "Images"), exist_ok=True)
            os.makedirs(os.path.join(path, "test_data", "Annotations"), exist_ok=True)
            # download train.txt
            file_url = os.path.join(url, "test_data", "train.txt")
            if verbose:
                print("Downloading filelist...")
            urlretrieve(file_url, os.path.join(path, "test_data", "train.txt"))
            # download image
            file_url = os.path.join(url, "test_data", "Images", "000040.jpg")
            if verbose:
                print("Downloading image...")
            urlretrieve(file_url, os.path.join(path, "test_data", "Images", "000040.jpg"))
            # download annotations
            file_url = os.path.join(url, "test_data", "Annotations", "000040.jpg.txt")
            if verbose:
                print("Downloading annotations...")
            urlretrieve(file_url, os.path.join(path, "test_data", "Annotations", "000040.jpg.txt"))

    def optimize(self, target_device):
        """This method is not used in this implementation."""
        return NotImplementedError

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    @staticmethod
    def __prepare_dataset(dataset, verbose=True):
        """
        This internal method prepares the train dataset depending on what type of dataset is provided.
        COCO is prepared according to: https://cv.gluon.ai/build/examples_datasets/mscoco.html

        If the dataset is of the DetectionDataset format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.

        :param dataset: the dataset
        :type dataset: ExternalDataset or DetectionDataset
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional

        :return: the modified dataset
        :rtype: VOCDetection, COCODetection or custom DetectionDataset depending on dataset argument
        """
        supported_datasets = ['coco', 'voc']
        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() not in supported_datasets:
                raise UserWarning("ExternalDataset dataset_type must be one of: ", supported_datasets)

            dataset_root = dataset.path

            if verbose:
                print("Loading {} type dataset...".format(dataset.dataset_type))

            if dataset.dataset_type.lower() == 'voc':
                from gluoncv.data import VOCDetection

                dataset = VOCDetection(root=dataset_root,
                                       splits=[(2007, 'trainval'), (2012, 'trainval')])

            elif dataset.dataset_type.lower() == 'coco':
                from gluoncv.data import COCODetection

                dataset = COCODetection(root=dataset_root,
                                        splits=['instances_train2017'])
            if verbose:
                print("ExternalDataset loaded.")
            return dataset
        elif isinstance(dataset, DetectionDataset) or issubclass(type(dataset), DetectionDataset):
            dataset.set_image_transform(ImageToNDArrayTransform())
            dataset.set_target_transform(BoundingBoxListToNumpyArray())
            return dataset
        else:
            raise ValueError("Dataset type {} not supported".format(type(dataset)))

    @staticmethod
    def __prepare_val_dataset(dataset, save_prefix='temp', data_shape=416, verbose=True):
        """
        This internal method prepares the train dataset depending on what type of dataset is provided.
        COCO is prepared according to: https://cv.gluon.ai/build/examples_datasets/mscoco.html

        If the dataset is of the DetectionDataset format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.

        :param dataset: the dataset
        :type dataset: ExternalDataset or DetectionDataset
        :param save_prefix: path where detections are stored temporarily for COCO dataset evaluation
        :type save_prefix: str, optional
        :param data_shape: data shape in pixels used for evaluation
        :type data_shape: int
        :param verbose: if True, additional information is printed on stdout
        :type verbose: bool, optional
        :return: the modified dataset
        :rtype: VOCDetection, COCODetection or custom DetectionDataset depending on dataset argument
        """

        supported_datasets = ['coco', 'voc']
        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() not in supported_datasets:
                raise UserWarning("dataset_type must be one of: ", supported_datasets)

            dataset_root = dataset.path

            if dataset.dataset_type.lower() == 'voc':
                from gluoncv.data import VOCDetection

                dataset = VOCDetection(root=dataset_root,
                                       splits=[(2007, 'test')])
                val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=dataset.classes)
                return dataset, val_metric
            elif dataset.dataset_type.lower() == 'coco':
                from gluoncv.data import COCODetection

                dataset = COCODetection(root=dataset_root, splits='instances_val2017',
                                        skip_empty=False)
                val_metric = COCODetectionMetric(
                    dataset, save_prefix + '_eval', cleanup=False, data_shape=(data_shape, data_shape))
                return dataset, val_metric

        elif isinstance(dataset, DetectionDataset) or issubclass(type(dataset), DetectionDataset):
            eval_metric = DetectionDatasetCOCOEval(dataset.classes, data_shape)
            dataset.set_image_transform(ImageToNDArrayTransform())
            dataset.set_target_transform(BoundingBoxListToNumpyArray())
            return dataset, eval_metric

        else:
            raise TypeError("Only ExternalDataset and DetectionDataset subclass types are supported.")
