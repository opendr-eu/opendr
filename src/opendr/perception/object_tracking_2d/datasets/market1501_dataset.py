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


import os
import cv2
import time
import numpy as np
from zipfile import ZipFile
from urllib.request import urlretrieve
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.data import Image
from opendr.engine.target import Category
from opendr.engine.datasets import ExternalDataset, DatasetIterator


class Market1501Dataset(ExternalDataset):
    def __init__(
        self,
        path,
    ):

        super().__init__(path, "market1501")

        self.path = path

    @staticmethod
    def download(
        url, download_path, dataset_sub_path=".", file_format="zip",
        create_dir=False,
    ):

        if file_format == "zip":
            if create_dir:
                os.makedirs(download_path, exist_ok=True)

            print("Downloading Market1501 Dataset zip file from", url, "to", download_path)

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

            print("Extracting Market1501 Dataset from zip file")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)

            os.remove(zip_path)

            return Market1501Dataset(
                os.path.join(download_path, dataset_sub_path),
            )
        else:
            raise ValueError("Unsupported file_format: " + file_format)

    @staticmethod
    def download_nano_market1501(
        download_path, create_dir=False,
    ):
        return Market1501Dataset.download(
            os.path.join(OPENDR_SERVER_URL, "perception", "object_tracking_2d", "nano_market1501.zip"),
            download_path,
            dataset_sub_path="nano_market1501",
            create_dir=create_dir,
        )


class Market1501DatasetIterator(DatasetIterator):

    def __init__(
        self,
        path,
    ):
        self.path = path
        self.files = os.listdir(self.path)

    def __getitem__(self, files_index):

        img, label = self.get_data(
            os.path.join(self.path, self.files[files_index]), self.files[files_index]
        )

        return (
            Image(img), Category(label)
        )

    def get_data(self, img_path, label_path):
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError("File corrupt {}".format(img_path))

        label = int(label_path.split("_")[0])

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        return img, label

    def __len__(self):
        return len(self.files)
