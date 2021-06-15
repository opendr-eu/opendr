# Copyright 2021 - present, OpenDR European Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# general imports
import os
import time
import json
import numpy as np
import warnings
from tqdm import tqdm
from urllib.request import urlretrieve

# gluoncv ssd imports
from gluoncv.data.transforms import presets
from gluoncv.data.batchify import Tuple, Stack, Pad
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from gluoncv import model_zoo
from gluoncv import utils as gutils
from gluoncv.loss import SSDMultiBoxLoss
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

gutils.random.seed(0)

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from opendr.engine.constants import OPENDR_SERVER_URL

# algorithm imports
from opendr.perception.object_detection_2d.utils import DetectionDatasetCOCOEval
from opendr.perception.object_detection_2d.datasets import DetectionDataset
from opendr.perception.object_detection_2d.datasets.transforms import ImageToNDArrayTransform, BoundingBoxListToNumpyArray, \
    transform_test


class SingleShotDetectorLearner(Learner):
    supported_backbones = {"vgg16_atrous": [512, 300],
                           "resnet50_v1": [512],
                           "mobilenet1.0": [512],
                           "mobilenet0.25": [300],
                           "resnet34_v1b": [300]}
    def __init__(self, lr=1e-3, epochs=120, batch_size=8,
                 device='cuda', backbone='vgg16_atrous',
                 img_size=512, lr_schedule='', temp_path='temp',
                 checkpoint_after_iter=5, checkpoint_load_iter=0,
                 val_after=5, log_after=100, num_workers=8,
                 weight_decay=5e-4, momentum=0.9):
        super(SingleShotDetectorLearner, self).__init__(lr=lr, batch_size=batch_size, lr_schedule=lr_schedule,
                                                        checkpoint_after_iter=checkpoint_after_iter,
                                                        checkpoint_load_iter=checkpoint_load_iter,
                                                        temp_path=temp_path, device=device, backbone=backbone)
        self.epochs = epochs
        self.log_after = log_after
        self.val_after = val_after
        self.num_workers = num_workers
        self.checkpoint_str_format = "checkpoint_epoch_{}.params"
        self.backbone = backbone.lower()

        if self.backbone not in self.supported_backbones:
            raise ValueError(self.backbone + " backbone is not supported. Call .info() function for a complete list of "
                                             "available backbones.")
        else:
            if img_size not in self.supported_backbones[self.backbone]:
                raise ValueError("Image size {} is not supported for the backbone."
                                 "Supported image sizes: ".format(img_size,self.backbone,
                                                                  self.supported_backbones[self.backbone]))

        if self.device == 'cuda':
            self.ctx = mx.gpu(0)
        else:
            self.ctx = mx.cpu()

        self.img_size = img_size
        self.weight_decay = weight_decay
        self.momentum = momentum

        self._model = None
        self.classes = None

        # Initialize temp path
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

    def info(self):
        print("The following backbone and image sizes are supported:")
        for k, v in self.supported_backbones.items():
            print('{}: {}'.format(k, v))

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

        os.makedirs(path, exist_ok=True)

        # model_name = 'ssd_' + self.backbone
        model_name = os.path.basename(path)
        metadata = {"model_paths": [], "framework": "mxnet", "format": "params",
                    "has_data": False, "inference_params": {}, "optimized": False,
                    "optimizer_info": {}, "backbone": self.backbone, "classes": self.classes}
        param_filepath = model_name + ".params"
        metadata["model_paths"].append(param_filepath)
        self._model.save_parameters(metadata["model_paths"][0])
        if verbose:
            print("Model parameters saved.")

        with open(os.path.join(path,  model_name + '.json'), 'w', encoding='utf-8') as f:
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
        if self._model is None:
            self.create_model(metadata["classes"])

        self._model.load_parameters(os.path.join(path, metadata["model_paths"][0]))
        self._model.collect_params().reset_ctx(self.ctx)
        self._model.hybridize(static_alloc=True, static_shape=True)
        if verbose:
            print("Loaded parameters and metadata.")
        return True

    def download(self, path=None, mode="pretrained", verbose=False,
                 # url=OPENDR_SERVER_URL + "perception/object_detection_2d/ssd/"):
                 url='ftp://155.207.131.93/perception/object_detection_2d/ssd/'):
        valid_modes = ["pretrained", "images"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":
            path = os.path.join(path, "ssd_default_person")
            if not os.path.exists(path):
                os.makedirs(path)

            if verbose:
                print("Downloading pretrained model...")

            file_url = os.path.join(url, "pretrained/ssd_default_person/ssd_default_person.json")
            if verbose:
                print("Downloading metadata...")
            urlretrieve(file_url, os.path.join(path, "ssd_default_person.json"))

            if verbose:
                print("Downloading params...")
            file_url = os.path.join(url, "pretrained/ssd_default_person/ssd_default_person.params")
            urlretrieve(file_url,
                        os.path.join(path, "ssd_default_person.params"))
        elif mode == "images":
            file_url = os.path.join(url, "images/people.jpg")
            if verbose:
                print("Downloading example image...")
            urlretrieve(file_url, os.path.join(path, "people.jpg"))


    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def optimize(self, target_device):
        """This method is not used in this implementation."""
        return NotImplementedError

    def create_model(self, classes):
        model_name = 'ssd_{}_{}_custom'.format(self.img_size, self.backbone)
        # self._model = model_zoo.get_model(model_name, classes=classes, pretrained_base=True)
        self._model = model_zoo.get_model(model_name, classes=classes, pretrained=True)
        self.classes = classes

    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        # NOTE: detection dataset must have dataset.dataset_type attribute
        save_prefix = 'ssd_{}_{}_{}'.format(self.img_size, self.backbone, dataset.dataset_type)

        # convert dataset to compatible format
        dataset = self.__prepare_dataset(dataset)

        # set save dir for checkpoint saving
        # NOTE: detection dataset must also have classes attribute
        self.create_model(dataset.classes)
        if verbose:
            print("Saving models as: {}".format(save_prefix))

        checkpoints_folder = os.path.join(self.temp_path, '{}_checkpoints'.format(save_prefix))
        if self.checkpoint_after_iter != 0 and not os.path.exists(checkpoints_folder):
            # user set checkpoint_after_iter so checkpoints must be created
            # create checkpoint dir
            os.makedirs(checkpoints_folder, exist_ok=True)

        start_epoch = 0
        if self.checkpoint_load_iter > 0:
            # user set checkpoint_load_iter, so load a checkpoint
            checkpoint_name = self.checkpoint_str_format.format(self.checkpoint_load_iter)
            checkpoint_path = os.path.join(checkpoints_folder, checkpoint_name)
            try:
                self._model.load_parameters(checkpoint_path)
                start_epoch = self.checkpoint_load_iter + 1
            except FileNotFoundError as e:
                e.strerror = 'No such file or directory {}'.format(checkpoint_path)

        # set device
        # NOTE: multi-gpu a little bugged
        if self.device == 'cuda':
            ctx = [mx.gpu(0)]
        else:
            ctx = [mx.cpu()]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._model.initialize()
            self._model.collect_params().reset_ctx(ctx[0])
        if verbose:
            print("Network:")
            print(self._model)

        # get data loader
        with autograd.train_mode():
            _, _, anchors = self._model(mx.nd.zeros((1, 3, self.img_size, self.img_size), ctx[0]))
        anchors = anchors.as_in_context(mx.cpu())

        # transform dataset & get loader
        train_transform = presets.ssd.SSDDefaultTrainTransform(self.img_size, self.img_size, anchors)
        dataset = dataset.transform(train_transform)

        batchify_fn = Tuple(Stack(), Stack(), Stack())
        train_loader = gluon.data.DataLoader(
            dataset, self.batch_size, shuffle=True, batchify_fn=batchify_fn,
            last_batch='rollover', num_workers=self.num_workers
        )

        trainer = gluon.Trainer(self._model.collect_params(),
                                'sgd', {'learning_rate': self.lr,
                                        'wd': self.weight_decay,
                                        'momentum': self.momentum},
                                update_on_kvstore=None)
        mbox_loss = SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('cross_entropy_loss')
        smoothl1_metric = mx.metric.Loss('smoothl1_loss')

        self._model.collect_params().reset_ctx(ctx)
        self._model.hybridize(static_alloc=True, static_shape=True)

        # start training
        training_dict = {"cross_entropy_loss": [], "smoothl1_loss": [], "val_map": []}
        n_iters = 0
        for epoch in range(start_epoch, self.epochs):
            autograd.set_training(True)
            cur_lr = self.__get_lr_at(epoch)
            trainer.set_learning_rate(cur_lr)

            self._model.hybridize(static_alloc=True, static_shape=True)

            tic = time.time()
            # TODO: epoch + 1
            print('[Epoch {}/{} lr={}]'.format(epoch, self.epochs, trainer.learning_rate))
            ce_metric.reset()
            smoothl1_metric.reset()

            for i, batch in enumerate(train_loader):
                n_iters += 1
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

                with autograd.record():
                    cls_preds = []
                    box_preds = []
                    for x in data:
                        cls_pred, box_pred, _ = self._model(x)
                        cls_preds.append(cls_pred)
                        box_preds.append(box_pred)
                    sum_loss, cls_loss, box_loss = mbox_loss(
                        cls_preds, box_preds, cls_targets, box_targets)
                    autograd.backward(sum_loss)

                trainer.step(1)

                ce_metric.update(0, [l * self.batch_size for l in cls_loss])
                smoothl1_metric.update(0, [l * self.batch_size for l in box_loss])
                if n_iters % self.log_after == self.log_after - 1:
                    name1, loss1 = ce_metric.get()
                    name2, loss2 = smoothl1_metric.get()
                    # TODO: epoch + 1
                    print('[Epoch {}][Batch {}] {}={:.3f}, {}={:.3f}'.format(
                        epoch, i, name1, loss1, name2, loss2
                    ))
            toc = time.time()

            # perform evaluation during training
            if epoch % self.val_after == self.val_after - 1 and val_dataset is not None:
                if verbose:
                    print("Model evaluation at epoch {}".format(epoch))
                eval_dict = self.eval(val_dataset)
                training_dict["val_map"].append(eval_dict["map"])

            # checkpoint saving
            if epoch % self.checkpoint_after_iter == self.checkpoint_after_iter - 1:
                if verbose:
                    print('Saving model at epoch {}'.format(epoch))
                checkpoint_name = self.checkpoint_str_format.format(epoch)
                checkpoint_filepath = os.path.join(checkpoints_folder, checkpoint_name)
                self.__save_checkpoint_at_epoch(checkpoint_filepath)

            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            training_dict["cross_entropy_loss"].append(loss1)
            training_dict["smoothl1_loss"].append(loss2)
            # TODO: epoch + 1
            print('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, toc - tic, name1, loss1, name2, loss2
            ))

        return training_dict

    def __save_checkpoint_at_epoch(self, filepath):
        self._model.save_parameters(filepath)

    def __get_lr_at(self, epoch):
        # TODO: explain lr_schedule options in init function doc
        if self.lr_schedule == '' or self.lr_schedule is None:
            return self.lr
        if self.lr_schedule == 'warmup':
            stop_epoch = max(3, int(0.03 * self.epochs))
            if epoch <= stop_epoch:
                return self.lr * (0.5 ** (stop_epoch - epoch))
            else:
                return self.lr
        else:
            return self.lr

    def eval(self, dataset):
        autograd.set_training(False)
        # NOTE: multi-gpu is a little bugged
        if self.device == 'cuda':
            ctx = [mx.gpu(0)]
        else:
            ctx = [mx.cpu()]
        print(self.device, ctx)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._model.initialize()
            self._model.collect_params().reset_ctx(ctx)
        self._model.hybridize(static_alloc=True, static_shape=True)
        self._model.set_nms(nms_thresh=0.45, nms_topk=400)

        dataset, eval_metric = self.__prepare_val_dataset(dataset, data_shape=self.img_size)

        eval_metric.reset()

        val_transform = presets.ssd.SSDDefaultValTransform(self.img_size, self.img_size)
        dataset = dataset.transform(val_transform)

        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            dataset, self.batch_size, shuffle=False, batchify_fn=val_batchify_fn, last_batch='keep',
            num_workers=self.num_workers)

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
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else np.zeros(ids.shape))

            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        map_name, mean_ap = eval_metric.get()

        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        print(val_msg)
        eval_dict = {k.lower(): v for k, v in zip(map_name, mean_ap)}
        return eval_dict

    def infer(self, img, threshold=0.2, keep_size=False):
        assert self._model is not None, "Model has not been loaded, call load(path) first"

        self._model.set_nms(nms_thresh=0.45, nms_topk=400)

        if isinstance(img, Image):
            _img = img.numpy()
        elif isinstance(img, np.ndarray):
            _img = img
        else:
            raise ValueError("Input should be of type Image or numpy array.")

        height, width, _ = _img.shape
        img_mx = mx.image.image.nd.from_numpy(_img)

        if keep_size:
            x, img_mx = transform_test(img_mx)
        else:
            x, img_mx = presets.ssd.transform_test(img_mx, short=self.img_size)

        h_mx, w_mx, _ = img_mx.shape
        x = x.as_in_context(self.ctx)
        class_IDs, scores, boxes = self._model(x)

        class_IDs = class_IDs[0, :, 0].asnumpy()
        scores = scores[0, :, 0].asnumpy()
        mask = np.where((class_IDs >= 0) & (scores > threshold))[0]
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

    @staticmethod
    def __prepare_dataset(dataset, verbose=True):
        """
        This internal method prepares the train dataset depending on what type of dataset is provided.
        COCO is prepared according to: https://cv.gluon.ai/build/examples_datasets/mscoco.html

        If the dataset is of the DatasetIterator format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.

        :param dataset: the dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param dataset_root: dataset root folder
        :type dataset_root: str
        :param verbose: whether to print additional information, defaults to 'True'
        :type verbose: bool, optional

        :raises UserWarning: UserWarnings with appropriate messages are raised for wrong type of dataset, or wrong paths
            and filenames

        :return: returns VOCDetection or COCODetection object or custom DatasetIterator implemented by user
        :rtype: VOCDetection or COCODetection class object or DatasetIterator instance
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
        elif isinstance(dataset, DatasetIterator) or issubclass(type(dataset), DatasetIterator):
            if issubclass(type(dataset), DetectionDataset):
                dataset.set_image_transform(ImageToNDArrayTransform())
                dataset.set_target_transform(BoundingBoxListToNumpyArray())
            return dataset
        else:
            raise ValueError("Dataset type {} not supported".format(type(dataset)))

    @staticmethod
    def __prepare_val_dataset(dataset, save_prefix='tmp', data_shape=512, verbose=True):
        """

        :param dataset:
        :type dataset:
        :param save_prefix:
        :type save_prefix:
        :param data_shape:
        :type data_shape:
        :param verbose:
        :type verbose:
        :return:
        :rtype:
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
        elif isinstance(dataset, DatasetIterator) or issubclass(type(dataset), DatasetIterator):
            # eval_metric = MeanAveragePrecision(dataset.classes, n_val_images=len(dataset))
            eval_metric = DetectionDatasetCOCOEval(dataset.classes, data_shape)
            if issubclass(type(dataset), DetectionDataset):
                dataset.set_image_transform(ImageToNDArrayTransform())
                dataset.set_target_transform(BoundingBoxListToNumpyArray())
            return dataset, eval_metric
        else:
            print("Dataset type {} not supported".format(type(dataset)))
            return dataset, None

if __name__ == '__main__':
    from opendr.perception.object_detection_2d.datasets import WiderPersonDataset
    from opendr.perception.object_detection_2d.datasets.transforms import ImageToNDArrayTransform, BoundingBoxListToNumpyArray
    dataset = WiderPersonDataset(root='/home/administrator/data/wider_person', splits=['train'],
                                 image_transform=ImageToNDArrayTransform(),
                                 target_transform=BoundingBoxListToNumpyArray()
                                 )

    ssd = SingleShotDetectorLearner(device="cuda", batch_size=6)
    # ssd.download(path=".", verbose=True)
    # ssd.load("ssd_default")

    # TODO: train
    ssd.fit(dataset)
