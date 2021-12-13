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
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.perception.object_detection_2d.datasets import DetectionDataset


class WiderPersonDataset(DetectionDataset):
    """
    WiderPerson dataset wrapper for OpenDR detectors. Assumes data has been downloaded from
    http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/ and unzipped in 'root' folder, so that it contains
    the 'Images' and 'Annotations' folders.
    """
    def __init__(self, root, splits, image_transform=None, target_transform=None,
                 transform=None):
        classes = ['person']
        self.root = root
        available_splits = ['train', 'val']
        self.splits = [split for split in splits if split in available_splits]
        self.image_dir = os.path.join(self.root, 'Images')
        self.anno_dir = os.path.join(self.root, 'Annotations')

        image_paths = []
        self.bboxes = []
        cls_id = 0
        for split in self.splits:
            with open(os.path.join(self.root, '{}.txt'.format(split))) as f:
                image_names = f.read().splitlines()

            for image_name in image_names:
                anno_file = os.path.join(self.anno_dir, image_name + '.jpg.txt')
                with open(anno_file) as f:
                    lines = f.readlines()
                    cur_line = 0

                    while True:
                        if len(lines) <= cur_line:
                            break

                        n_boxes = max(1, int(lines[cur_line][:-1]))
                        bboxes = lines[cur_line + 1:cur_line + n_boxes + 1]
                        bounding_boxes = []
                        for bbox in bboxes:
                            bbox = bbox.split(' ')
                            # convert to (xmin, ymin, xmax, ymax, c) format
                            # TODO: use BoundingBoxList format?
                            # coord = np.asarray([float(bbox[1]), float(bbox[2]),
                            #                     float(bbox[3]), float(bbox[4]), int(cls_id)])
                            bounding_box = BoundingBox(name=int(cls_id),
                                                       left=float(bbox[1]), top=float(bbox[2]),
                                                       width=float(bbox[3]) - float(bbox[1]),
                                                       height=float(bbox[4]) - float(bbox[2]))
                            # skip box if it's too small
                            # w = coord[2] - coord[0]
                            w = bounding_box.width
                            h = bounding_box.height
                            # h = coord[3] - coord[1]
                            if min(w, h) < 64:
                                continue
                            bounding_boxes.append(bounding_box)
                        if bounding_boxes:
                            # self.bboxes.append(np.asarray(bounding_boxes))
                            self.bboxes.append(BoundingBoxList(boxes=bounding_boxes))
                            image_paths.append(os.path.join(self.image_dir, image_name + '.jpg'))
                        cur_line += 2 + n_boxes
        dataset_type = 'wider_person'
        super().__init__(classes=classes, dataset_type=dataset_type, image_paths=image_paths,
                         image_transform=image_transform, target_transform=target_transform,
                         transform=transform, splits=splits, root=root)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.bboxes[item]
        # read image, apply transform, return result
        img = cv2.imread(image_path)
        if self._image_transform is not None:
            img = self._image_transform(img)

        if self._target_transform is not None:
            label = self._target_transform(label)

        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def get_image(self, item):
        image_path = self.image_paths[item]
        img = cv2.imread(image_path)
        if self._image_transform is not None:
            img = self._image_transform(img)
        return img

    def get_bboxes(self, item):
        boxes = self.bboxes[item]
        if self._target_transform is not None:
            boxes = self._target_transform(boxes)
        return boxes

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    from opendr.perception.object_detection_2d.utils.vis_utils import draw_bounding_boxes

    dataset = WiderPersonDataset('/home/administrator/data/wider_person',
                                 splits=['train'])
    print(len(dataset))

    all_boxes = [[[] for _ in range(len(dataset))]
                 for _ in range(dataset.num_classes)]

    for i, (img, targets) in enumerate(dataset):
        img = draw_bounding_boxes(img, targets, class_names=dataset.classes)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
