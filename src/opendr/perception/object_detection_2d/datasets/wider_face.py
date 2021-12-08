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

import numpy as np
import cv2
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.perception.object_detection_2d.datasets.detection_dataset import DetectionDataset


class WiderFaceDataset(DetectionDataset):
    """
    WiderFace dataset wrapper for OpenDR face detectors. Assumes data has been downloaded from
    http://shuoyang1213.me/WIDERFACE/ and unzipped in 'root' folder, so that it contains
    the 'WIDER_train', 'WIDER_val', 'WIDER_test' and 'wider_face_split' folders.
    """
    def __init__(self, root, splits, ignore_tiny=True, image_transform=None, target_transform=None, transform=None):
        classes = ['face']
        self.root = root
        available_splits = ['train', 'val', 'test']
        self.splits = [split for split in splits if split in available_splits]
        self.anno_dir = annotations_dir = os.path.join(self.root, 'wider_face_split')

        image_paths = []
        self.bboxes = []
        cls_id = 0
        for split in self.splits:
            if split != 'test':
                annotations_file = os.path.join(annotations_dir, 'wider_face_{}_bbx_gt.txt'.format(split))
                with open(annotations_file) as f:
                    lines = f.readlines()
                    cur_line = 0

                    while True:
                        if len(lines) == cur_line:
                            break

                        image_name = lines[cur_line][:-1]
                        n_faces = max(1, int(lines[cur_line + 1]))
                        bboxes = lines[cur_line + 2:cur_line + n_faces + 2]
                        bboxes_list = []
                        for bbox in bboxes:
                            bbox = bbox.split(' ')
                            # check invalid
                            if int(bbox[7]) == 1:
                                continue
                            bbox_odr = BoundingBox(name=cls_id, left=float(bbox[0]), top=float(bbox[1]),
                                                   width=float(bbox[2]), height=float(bbox[3]))
                            if ignore_tiny and max(bbox_odr.width, bbox_odr.height) < 10:
                                continue
                            bboxes_list.append(bbox_odr)
                        cur_line += 2 + n_faces
                        if bboxes_list:
                            self.bboxes.append(BoundingBoxList(boxes=bboxes_list))
                            image_dir = os.path.join(self.root, 'WIDER_{}'.format(split), 'images')
                            image_paths.append(os.path.join(image_dir, image_name))
            else:
                # test split, only images are available
                image_dir = os.path.join(self.root, 'WIDER_{}'.format(split), 'images')
                subdirs_names = [subdir for subdir in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, subdir))]
                print(subdirs_names)
                image_paths = []
                for subdir in subdirs_names:
                    image_names = os.listdir(os.path.join(image_dir, subdir))
                    image_paths.extend([os.path.join(image_dir, subdir, image_name) for image_name in image_names])
                self.bboxes = [np.empty((1, 5)) for _ in range(len(image_paths))]

        dataset_type = 'wider_face'
        super().__init__(classes=classes, dataset_type=dataset_type, image_paths=image_paths,
                         image_transform=image_transform, target_transform=target_transform,
                         transform=transform, splits=splits, root=root)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.bboxes[item]

        img = cv2.imread(image_path)
        if self._image_transform is not None:
            img = self._image_transform(img)

        if self._target_transform is not None:
            label = self._target_transform(label)

        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def pull_image(self, item):
        image_path = self.image_paths[item]
        img = cv2.imread(image_path)
        return img

    def get_bboxes(self, item):
        return self.bboxes[item]

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    from opendr.perception.object_detection_2d.utils.vis_utils import draw_bounding_boxes

    dataset = WiderFaceDataset('/home/administrator/data/wider', splits=['train', 'val'])

    all_boxes = [[[] for _ in range(len(dataset))]
                 for _ in range(dataset.num_classes)]

    for i, (img, targets) in enumerate(dataset):
        img = draw_bounding_boxes(img, targets, class_names=dataset.classes)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
