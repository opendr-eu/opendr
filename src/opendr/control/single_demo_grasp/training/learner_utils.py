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

from detectron2.structures import BoxMode


def get_train_dicts(img_dir, bbx_train, kps_train, num_train):

    dataset_dicts = []
    for idx in range(0, num_train):
        record = {}

        filename = img_dir + "/" + str(idx) + ".jpg"
        height = 480
        width = 640

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        kps = []
        for i in range(len(kps_train[idx][0])):
            kps.append(kps_train[idx][0][i])
            kps.append(kps_train[idx][1][i])
            kps.append(2)  # visibility

        objs = []
        obj = {
            "bbox": [bbx_train[idx][0], bbx_train[idx][1], bbx_train[idx][2], bbx_train[idx][3]],
            "bbox_mode": BoxMode.XYXY_ABS,
            "keypoints": kps,  # x-y-visibility
            "category_id": 0,
            "iscrowd": 0
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def get_val_dicts(img_dir, bbx_val, kps_val, num_val):

    dataset_dicts = []
    for idx in range(0, num_val):
        record = {}

        filename = img_dir + "/" + str(idx) + ".jpg"
        height = 480
        width = 640

        kps = []
        for i in range(len(kps_val[idx][0])):
            kps.append(kps_val[idx][0][i])
            kps.append(kps_val[idx][1][i])
            kps.append(2)  # visibility

        record["file_name"] = filename
        record["height"] = height
        record["width"] = width

        objs = []
        obj = {
            "bbox": [bbx_val[idx][0], bbx_val[idx][1], bbx_val[idx][2], bbx_val[idx][3]],
            "bbox_mode": BoxMode.XYXY_ABS,
            "keypoints": kps,  # x-y-visibility
            "category_id": 0,
            "iscrowd": 0
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_datasets(DatasetCatalog, MetadataCatalog, image_dir, object_name,
                      bbx_train, kps_train, bbx_val, kps_val):

    num_train = len(bbx_train)
    num_val = len(bbx_val)
    num_kps = len(kps_train[0][0])
    kps_names = []

    for i in range(num_kps):
        kps_names.append("p" + str(i + 1))

    for d in ["train"]:

        DatasetCatalog.register(object_name + "_" + d, lambda d=d: get_train_dicts(
                                    image_dir + "/" + object_name + "/images/" + d, bbx_train, kps_train, num_train))

        MetadataCatalog.get(object_name + "_" + d).set(thing_classes=[object_name])
        MetadataCatalog.get(object_name + "_" + d).set(keypoint_names=kps_names)
        MetadataCatalog.get(object_name + "_" + d).set(keypoint_flip_map=[])

    train_set = get_train_dicts(image_dir + "/" + object_name + "/images/" + d, bbx_train, kps_train, num_train)

    for d in ["val"]:

        DatasetCatalog.register(object_name + "_" + d, lambda d=d: get_val_dicts(
                                    image_dir + "/" + object_name + "/images/" + d, bbx_val, kps_val, num_val))

        MetadataCatalog.get(object_name + "_" + d).set(thing_classes=[object_name])
        MetadataCatalog.get(object_name + "_" + d).set(keypoint_names=kps_names)
        MetadataCatalog.get(object_name + "_" + d).set(keypoint_flip_map=[])

    val_set = get_val_dicts(image_dir + "/" + object_name + "/images/" + d, bbx_val, kps_val, num_val)

    return MetadataCatalog.get(object_name + "_" + "train"), train_set, val_set
