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
#
# JointDataset, DetDataset, LoadImages, LoadVideo, LoadImagesAndLabels and splits folder are taken from FairMOT

import glob
import math
import os
import os.path as osp
import random
import time
import cv2
import copy
import numpy as np
import torch
from collections import OrderedDict
from urllib.request import urlretrieve
from zipfile import ZipFile
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.data import Image, ImageWithDetections
from opendr.engine.target import TrackingAnnotationList
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from torchvision.transforms import transforms as T
from opendr.perception.object_tracking_2d.fair_mot.algorithm.gen_labels_mot import gen_labels_mot


class MotDataset(ExternalDataset):
    def __init__(
        self,
        path,
    ):

        super().__init__(path, "mot")

        self.path = path

        self.__prepare_dataset()

    @staticmethod
    def download(
        url, download_path, dataset_sub_path=".", file_format="zip",
        create_dir=False,
    ):

        if file_format == "zip":
            if create_dir:
                os.makedirs(download_path, exist_ok=True)

            print("Downloading MOT20 Dataset zip file from", url, "to", download_path)

            start_time = 0
            last_print = 0

            def reporthook(count, block_size, total_size):
                nonlocal start_time
                nonlocal last_print
                if count == 0:
                    start_time = time.time()
                    last_print = start_time
                    return

                duration = time.time() - start_time
                progress_size = int(count * block_size)
                speed = int(progress_size / (1024 * duration))
                if time.time() - last_print >= 1:
                    last_print = time.time()
                    print(
                        "\r%d MB, %d KB/s, %d seconds passed" %
                        (progress_size / (1024 * 1024), speed, duration),
                        end=''
                    )

            zip_path = os.path.join(download_path, "dataset.zip")
            urlretrieve(url, zip_path, reporthook=reporthook)
            print()

            print("Extracting MOT20 Dataset from zip file")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)

            os.remove(zip_path)

            return MotDataset(
                os.path.join(download_path, dataset_sub_path),
            )
        else:
            raise ValueError("Unsupported file_format: " + file_format)

    @staticmethod
    def download_nano_mot20(
        download_path, create_dir=False,
    ):
        return MotDataset.download(
            os.path.join(OPENDR_SERVER_URL, "perception", "object_tracking_2d", "nano_MOT20.zip"),
            download_path,
            create_dir=create_dir,
            dataset_sub_path=".",
        )

    def __prepare_dataset(self):

        datasets = os.listdir(self.path)

        for dataset in datasets:
            seq_root = os.path.join(self.path, dataset, "images/train")
            label_root = os.path.join(self.path, dataset, "labels_with_ids/train")

            if not os.path.exists(label_root):
                gen_labels_mot(seq_root, label_root)


class RawMotDatasetIterator(DatasetIterator):
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(
        self,
        root,
        paths,
        down_ratio=4,
        max_objects=500,
        ltrb=True,
        mse_loss=False,
        img_size=(1088, 608),
        scan_labels=True,
    ):
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        for ds, path in paths.items():
            with open(path, "r") as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    osp.join(root, x.strip()) for x in self.img_files[ds]
                ]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds])
                )

            self.label_files[ds] = [
                x.replace("images", "labels_with_ids")
                .replace(".png", ".txt")
                .replace(".jpg", ".txt")
                for x in self.img_files[ds]
            ]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            if scan_labels:
                for lp in label_paths:
                    lb = np.loadtxt(lp)
                    if len(lb) < 1:
                        continue
                    if len(lb.shape) < 2:
                        img_max = lb[1]
                    else:
                        img_max = np.max(lb[:, 1])
                    if img_max > max_index:
                        max_index = img_max
            else:
                max_index = 500
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = max_objects
        self.down_ratio = down_ratio
        self.ltrb = ltrb
        self.mse_loss = mse_loss

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(
            img_path, label_path
        )

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return (
            Image(imgs), TrackingAnnotationList.from_mot(labels)
        )

    def get_data(self, img_path, label_path):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError("File corrupt {}".format(img_path))

        h, w, _ = img.shape
        shape = img.shape[:2]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (
            round(shape[1] * ratio),
            round(shape[0] * ratio),
        )
        padw = (width - new_shape[0]) / 2  # width padding
        padh = (height - new_shape[1]) / 2  # height padding

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = (
                ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            )
            labels[:, 3] = (
                ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            )
            labels[:, 4] = (
                ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            )
            labels[:, 5] = (
                ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
            )
        else:
            labels = np.array([])

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height

        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR to RGB

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


class RawMotWithDetectionsDatasetIterator(DatasetIterator):
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(
        self,
        root,
        paths,
        img_size=(1088, 608),
    ):
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        for ds, path in paths.items():
            with open(path, "r") as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    osp.join(root, x.strip()) for x in self.img_files[ds]
                ]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds])
                )

            self.label_files[ds] = [
                x.replace("images", "labels_with_ids")
                .replace(".png", ".txt")
                .replace(".jpg", ".txt")
                for x in self.img_files[ds]
            ]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(
            img_path, label_path
        )

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        target = TrackingAnnotationList.from_mot(labels)
        detections = target.bounding_box_list()
        data = ImageWithDetections(imgs, detections)

        return (
            data, target
        )

    def get_data(self, img_path, label_path):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError("File corrupt {}".format(img_path))

        h, w, _ = img.shape
        shape = img.shape[:2]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (
            round(shape[1] * ratio),
            round(shape[0] * ratio),
        )
        padw = (width - new_shape[0]) / 2  # width padding
        padh = (height - new_shape[1]) / 2  # height padding

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = (
                ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            )
            labels[:, 3] = (
                ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            )
            labels[:, 4] = (
                ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            )
            labels[:, 5] = (
                ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
            )
        else:
            labels = np.array([])

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height

        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR to RGB

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


class MotDatasetIterator(DatasetIterator):
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(
        self,
        root,
        paths,
        down_ratio=4,
        max_objects=500,
        ltrb=True,
        mse_loss=False,
        img_size=(1088, 608),
        augment=False,
        transforms=T.Compose([T.ToTensor()]),
    ):
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        for ds, path in paths.items():
            with open(path, "r") as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    osp.join(root, x.strip()) for x in self.img_files[ds]
                ]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds])
                )

            self.label_files[ds] = [
                x.replace("images", "labels_with_ids")
                .replace(".png", ".txt")
                .replace(".jpg", ".txt")
                for x in self.img_files[ds]
            ]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = max_objects
        self.down_ratio = down_ratio
        self.ltrb = ltrb
        self.mse_loss = mse_loss
        self.augment = augment
        self.transforms = transforms

        print("=" * 80)
        print("dataset iterator summary")
        print(self.tid_num)
        print("total # identities:", self.nID)
        print("start index")
        print(self.tid_start_index)
        print("=" * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(
            img_path, label_path
        )

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return (
            Image(imgs, dtype=np.float32), TrackingAnnotationList.from_mot(labels)
        )

    def get_data(self, img_path, label_path):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError("File corrupt {}".format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = (
                ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            )
            labels[:, 3] = (
                ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            )
            labels[:, 4] = (
                ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            )
            labels[:, 5] = (
                ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
            )
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(
                img,
                labels,
                degrees=(-5, 5),
                translate=(0.10, 0.10),
                scale=(0.50, 1.20),
            )

        plotFlag = False
        if plotFlag:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(
                labels[:, [1, 3, 3, 1, 1]].T,
                labels[:, [2, 2, 4, 4, 2]].T,
                ".-",
            )
            plt.axis("off")
            plt.savefig("test.jpg")
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = [".jpg", ".jpeg", ".png", ".tif"]
            self.files = sorted(glob.glob("%s/*.*" % path))
            self.files = list(
                filter(
                    lambda x: os.path.splitext(x)[1].lower() in image_format,
                    self.files,
                )
            )
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, "No images found in " + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, "Failed to load " + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, "Failed to load " + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print("Lenth of the video: {:d} frames".format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, "Failed to load frame {:d}".format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(
        self, path, img_size=(1088, 608), augment=False, transforms=None
    ):
        with open(path, "r") as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace("\n", "") for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [
            x.replace("images", "labels_with_ids")
            .replace(".png", ".txt")
            .replace(".jpg", ".txt")
            for x in self.img_files
        ]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError("File corrupt {}".format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = (
                ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            )
            labels[:, 3] = (
                ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            )
            labels[:, 4] = (
                ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            )
            labels[:, 5] = (
                ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
            )
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(
                img,
                labels,
                degrees=(-5, 5),
                translate=(0.10, 0.10),
                scale=(0.50, 1.20),
            )

        plotFlag = False
        if plotFlag:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(
                labels[:, [1, 3, 3, 1, 1]].T,
                labels[:, [2, 2, 4, 4, 2]].T,
                ".-",
            )
            plt.axis("off")
            plt.savefig("test.jpg")
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(
    img, height=608, width=1088, color=(127.5, 127.5, 127.5)
):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(
        img, new_shape, interpolation=cv2.INTER_AREA
    )  # resized, no border
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # padded rectangular
    return img, ratio, dw, dh


def random_affine(
    img,
    targets=None,
    degrees=(-10, 10),
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
    shear=(-2, 2),
    borderValue=(127.5, 127.5, 127.5),
):
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(
        angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s
    )

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[
        0
    ] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[
        1
    ] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # x shear (deg)
    S[1, 0] = math.tan(
        (random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180
    )  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(
        img,
        M,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=borderValue,
    )  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (
                points[:, 3] - points[:, 1]
            )

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
                .reshape(4, n)
                .T
            )

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = (
                max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            )
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = (
                np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                .reshape(4, n)
                .T
            )

            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, M
    else:
        return imw


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(
        self,
        root,
        paths,
        down_ratio=4,
        max_objects=500,
        ltrb=True,
        mse_loss=False,
        img_size=(1088, 608),
        augment=False,
        transforms=None,
    ):
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        for ds, path in paths.items():
            with open(path, "r") as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    osp.join(root, x.strip()) for x in self.img_files[ds]
                ]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds])
                )

            self.label_files[ds] = [
                x.replace("images", "labels_with_ids")
                .replace(".png", ".txt")
                .replace(".jpg", ".txt")
                for x in self.img_files[ds]
            ]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = max_objects
        self.down_ratio = down_ratio
        self.ltrb = ltrb
        self.mse_loss = mse_loss
        self.augment = augment
        self.transforms = transforms

        print("=" * 80)
        print("dataset summary")
        print(self.tid_num)
        print("total # identities:", self.nID)
        print("start index")
        print(self.tid_start_index)
        print("=" * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(
            img_path, label_path
        )

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return process(
            Image(imgs, dtype=np.float32), TrackingAnnotationList.from_mot(labels),
            self.ltrb, self.down_ratio,
            self.max_objs, self.num_classes, self.mse_loss
        )


class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(
        self, root, paths, img_size=(1088, 608), augment=False, transforms=None
    ):

        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, "r") as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    osp.join(root, x.strip()) for x in self.img_files[ds]
                ]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds])
                )

            self.label_files[ds] = [
                x.replace("images", "labels_with_ids")
                .replace(".png", ".txt")
                .replace(".jpg", ".txt")
                for x in self.img_files[ds]
            ]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print("=" * 80)
        print("dataset summary")
        print(self.tid_num)
        print("total # identities:", self.nID)
        print("start index")
        print(self.tid_start_index)
        print("=" * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs, labels0, img_path, (h, w)


def process(imgs, labels, ltrb, down_ratio, max_objs, num_classes, mse_loss):

    imgs = imgs.numpy()
    labels = labels.mot(with_confidence=False)

    output_h = imgs.shape[1] // down_ratio
    output_w = imgs.shape[2] // down_ratio
    num_classes = num_classes
    num_objs = labels.shape[0]
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    if ltrb:
        wh = np.zeros((max_objs, 4), dtype=np.float32)
    else:
        wh = np.zeros((max_objs, 2), dtype=np.float32)
    reg = np.zeros((max_objs, 2), dtype=np.float32)
    ind = np.zeros((max_objs,), dtype=np.int64)
    reg_mask = np.zeros((max_objs,), dtype=np.uint8)
    ids = np.zeros((max_objs,), dtype=np.int64)
    bbox_xys = np.zeros((max_objs, 4), dtype=np.float32)

    draw_gaussian = (
        draw_msra_gaussian if mse_loss else draw_umich_gaussian
    )
    for k in range(num_objs):
        label = labels[k]
        bbox = label[2:]
        cls_id = int(label[0])
        bbox[[0, 2]] = bbox[[0, 2]] * output_w
        bbox[[1, 3]] = bbox[[1, 3]] * output_h
        bbox_amodal = copy.deepcopy(bbox)
        bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.0
        bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.0
        bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
        bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
        bbox[0] = np.clip(bbox[0], 0, output_w - 1)
        bbox[1] = np.clip(bbox[1], 0, output_h - 1)
        h = bbox[3]
        w = bbox[2]

        bbox_xy = copy.deepcopy(bbox)
        bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
        bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
        bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
        bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            radius = 6 if mse_loss else radius
            ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_gaussian(hm[cls_id], ct_int, radius)
            if ltrb:
                wh[k] = (
                    ct[0] - bbox_amodal[0],
                    ct[1] - bbox_amodal[1],
                    bbox_amodal[2] - ct[0],
                    bbox_amodal[3] - ct[1],
                )
            else:
                wh[k] = 1.0 * w, 1.0 * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            ids[k] = label[1]
            bbox_xys[k] = bbox_xy

    ret = {
        "input": imgs,
        "hm": hm,
        "reg_mask": reg_mask,
        "ind": ind,
        "wh": wh,
        "reg": reg,
        "ids": ids,
        "bbox": bbox_xys,
    }
    return ret


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]
    masked_gaussian = gaussian[
        radius - top: radius + bottom, radius - left: radius + right
    ]
    if (
        min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0
    ):  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = (
        np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) *
        value
    )
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]
    masked_regmap = regmap[:, y - top: y + bottom, x - left: x + right]
    masked_gaussian = gaussian[
        radius - top: radius + bottom, radius - left: radius + right
    ]
    masked_reg = reg[
        :, radius - top: radius + bottom, radius - left: radius + right
    ]
    if (
        min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0
    ):  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1]
        )
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top: y + bottom, x - left: x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]: img_y[1], img_x[0]: img_x[1]] = np.maximum(
        heatmap[img_y[0]: img_y[1], img_x[0]: img_x[1]],
        g[g_y[0]: g_y[1], g_x[0]: g_x[1]],
    )
    return heatmap


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y
