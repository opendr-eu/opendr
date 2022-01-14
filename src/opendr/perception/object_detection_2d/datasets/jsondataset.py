# Copyright 2020-2021 OpenDR European Project
#
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
from pycocotools.coco import COCO

from opendr.engine.data import Image
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.perception.object_detection_2d.datasets.detection_dataset import DetectionDataset, is_image_type, remove_extension


class JSONBasedDataset(DetectionDataset):
    """
    Reader class for datasets annotated with the LabelImg tool in Pascal VOC XML format.
    The dataset should be in the following structure:
    data_root
    |-- images
    |-- annotations
    The exact names of the folders can be passed as arguments (images_dir) and (annotations_dir).
    """
    def __init__(self, dataset_type, root, classes=None, image_transform=None,
                 target_transform=None, transform=None, splits='',
                 images_dir='images', annotations_dir='annotations', json_file='instances_val2017.json', iscrowd=None, preload_anno=False):
        self.abs_images_dir = os.path.join(root, images_dir)
        self.abs_annot_dir = os.path.join(root, annotations_dir)
        self.json_file = json_file
        self.iscrowd=iscrowd
        image_names = [im_filename for im_filename in os.listdir(self.abs_images_dir)
                       if is_image_type(im_filename)]

        if classes is None:
            classes = []
        self.classes = classes
        super().__init__(classes, dataset_type, root, image_transform=image_transform, target_transform=target_transform,
                         transform=transform, image_paths=image_names, splits=splits)
        self.bboxes = []
        self.preload_anno = preload_anno
        annot_file = os.path.join(self.abs_annot_dir, json_file)
        #with open(annot_file, 'rb') as file:
        #    doc = json.load(file)
        coco = COCO(os.path.join(annot_file))
        catIds = coco.getCatIds(catNms=classes)
        self.coco = coco
        self.catIds = catIds
        ##TODO make a list of all categories in coco
        #df = pd.read_csv('/home/manos/PycharmProjects/downLoadCoco/coco_categories.csv')
        #df.set_index('id', inplace=True)
        #self.df = df

        #annotations = doc['annotations']
        #annotations = coco.loadAnns(coco.getAnnIds())
        #self.annotations = annotations

        if preload_anno:
            for image_name in image_names:
                image_id = int(remove_extension(image_name))
                if classes:
                    annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=catIds, iscrowd=iscrowd))
                else:
                    annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=iscrowd))

                bboxes = self._read_annotation_file(annotations)
                self.bboxes.append(bboxes)

    def _read_annotation_file(self, anno):

        #category = self.df.loc[anno['category_id']]['name']
        boxes = [an['bbox'] for an in anno]
        class_ids = [an['category_id'] for an in anno]
        cats = self.coco.loadCats(class_ids)
        classes = [cat['name'] for cat in cats]


        #objs = ['person', boxes[0], boxes[1], boxes[0 ] +boxes[2], boxes[1 ] +boxes[3]]
        bounding_boxes = []
        for box,clas in zip(boxes, classes):

            cls_name = clas
            if cls_name not in self.classes:
                self.classes.append(cls_name)

            cls_id = self.classes.index(cls_name)

            xmin = (float(box[0]) - 1)
            ymin = (float(box[1]) - 1)
            xmax = (float(box[0]+box[2]) - 1)
            ymax = (float(box[1]+box[3]) - 1)
            # label.append([xmin, ymin, xmax, ymax, cls_id])
            bounding_box = BoundingBox(name=int(cls_id),
                                       left=float(xmin), top=float(ymin),
                                       width=float(xmax) - float(xmin),
                                       height=float(ymax) - float(ymin))
            bounding_boxes.append(bounding_box)
        return BoundingBoxList(boxes=bounding_boxes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_name = self.image_paths[item]
        image_path = os.path.join(self.abs_images_dir, image_name)
        img_np = cv2.imread(image_path)
        img = Image(img_np)

        if self.preload_anno:
            label = self.bboxes[item]
        else:
            image_id = int(remove_extension(image_name))
            #anno = self.annotations[next((i for i, x in enumerate(self.annotations) if x["image_id"] == image_id), None)]
            if self.classes:
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, catIds=self.catIds, iscrowd=self.iscrowd))
            else:
                annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id, iscrowd=self.iscrowd))

            label = self._read_annotation_file(annotations)

        if self._image_transform is not None:
            img = self._image_transform(img)

        if self._target_transform is not None:
            label = self._target_transform(label)

        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def get_image(self, item):
        image_name = self.image_paths[item]
        image_path = os.path.join(self.abs_images_dir, image_name)
        img_np = cv2.imread(image_path)
        if self._image_transform is not None:
            img = self._image_transform(img_np)
        return img

    def get_bboxes(self, item):
        boxes = self.bboxes[item]
        if self._target_transform is not None:
            boxes = self._target_transform(boxes)
        return boxes


if __name__ == '__main__':
    from opendr.perception.object_detection_2d.utils.vis_utils import draw_bounding_boxes

    dataset = JSONBasedDataset(root='/home/manos/data/coco2017/', dataset_type='coco',
                               images_dir='val2017', annotations_dir='annotations',
                               json_file='instances_val2017.json')#, classes=['person'])

    print(len(dataset))

    all_boxes = [[[] for _ in range(len(dataset))]
                 for _ in range(dataset.num_classes)]

    for i, (img, targets) in enumerate(dataset):
        img = draw_bounding_boxes(img, targets, class_names=dataset.classes)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
