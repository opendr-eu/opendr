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

import json
import multiprocessing as mp
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Tuple, Any, Dict, Union, List

import numpy as np
from PIL import Image as PilImage
from cityscapesscripts.evaluation.evalPanopticSemanticLabeling import pq_compute_multi_core, average_pq
from cityscapesscripts.helpers.labels import labels as cs_labels
from cityscapesscripts.preparation.createPanopticImgs import convert2panoptic
from mmdet.datasets import CityscapesDataset as MmdetCityscapesDataset
from mmdet.datasets import build_dataset
from pycococreatortools import pycococreatortools as pct
from tqdm import tqdm

from opendr.engine.data import Image
from opendr.engine.datasets import ExternalDataset, DatasetIterator


class CityscapesDataset(ExternalDataset, DatasetIterator):
    """
    The CityscapesDataset class provides the OpenDR and mmdet APIs for different use cases. Inference using pre-trained
    models is supported by the OpenDR interface, for training and evaluation an instance of the respective mmdet version
    is created.

    Use the static method prepare_data() to convert the raw Cityscapes dataset to the structure below.

    The folder structure should look like this:
    path
    ├── annotations.json
    ├── panoptic_gt.json  [only required for evaluation]
    ├── images
    │   ├── img_0.png
    │   └── ...
    ├── panoptic_gt       [only required for evaluation]
    │   ├── img_0_gtFine_panoptic.png
    │   └── ...
    └── stuffthingmaps
        ├── img_0.png
        └── ...
    """

    def __init__(self, path: str):
        """
        :param path: path to the top level directory of the dataset
        :type path: str
        """
        super().__init__(path=path, dataset_type='cityscapes')

        self._image_folder = Path(self.path) / 'images'
        self._segmentation_folder = Path(self.path) / 'stuffthingmaps'
        self._annotation_file = Path(self.path) / 'annotations.json'
        self._panoptic_gt_folder = Path(self.path) / 'panoptic_gt'
        self._panoptic_gt_file = Path(self.path) / 'panoptic_gt.json'

        self._image_filenames = sorted([f for f in self._image_folder.glob('*') if f.is_file()])
        self._segmentation_filenames = sorted([f for f in self._segmentation_folder.glob('*') if f.is_file()])
        for img, seg in zip(self._image_filenames, self._segmentation_filenames):
            assert img.name == seg.name

        # Used for evaluation
        if self._panoptic_gt_folder.exists():
            assert self._panoptic_gt_file.exists()
            self._panoptic_gt_filenames = sorted([f for f in self._panoptic_gt_folder.glob('*') if f.is_file()])
            for img, gt in zip(self._image_filenames, self._panoptic_gt_filenames):
                assert img.name == gt.name.replace('_gtFine_panoptic', '')

        self._pipeline = None
        self._mmdet_dataset = (None, None)  # (object, test_mode)

    def get_mmdet_dataset(self,
                          test_mode: bool = False
                          ) -> MmdetCityscapesDataset:
        """
        Returns the dataset in a format compatible with the mmdet dataloader.

        :param test_mode: if set to True, the panoptic ground truth data has to be present
        :type test_mode: bool
        """
        if self._mmdet_dataset[0] is None or self._mmdet_dataset[1] != test_mode:
            self._mmdet_dataset = (self._build_mmdet_dataset(test_mode), test_mode)
        return self._mmdet_dataset[0]

    def evaluate(self,
                 prediction_path: Union[Path, str],
                 prediction_json_folder: Union[Path, str]
                 ) -> Dict[str, Any]:
        """
        This method is used to evaluate the predictions versus the ground truth returns the following stats:
            - Panoptic Quality (PQ)
            - Segmentation Quality (SQ)
            - Recognition Quality (RQ)

        This function contains modified code from '_evaluate_panoptic()' in
            src/opendr/perception/panoptic_segmentation/efficient_ps/src/mmdetection/mmdet/datasets/cityscapes.py

        :param prediction_path: path to the predicted stuffandthing maps
        :type prediction_path: str, pathlib.Path
        :param prediction_json_folder: path to the predicted annotations
        :type prediction_json_folder: str, pathlib.Path
        :return: returns evaluation stats
        :rtype: dict
        """
        if isinstance(prediction_path, str):
            prediction_path = Path(prediction_path)
        if isinstance(prediction_json_folder, str):
            prediction_json_folder = Path(prediction_json_folder)

        if not prediction_path.exists():
            raise ValueError('The provided prediction_path does not exist.')
        if not prediction_json_folder.exists():
            raise ValueError('The provided prediction_json_folder does not exist.')

        with open(self._panoptic_gt_file, 'r') as f:
            gt_json = json.load(f)
        categories = {el['id']: el for el in gt_json['categories']}

        pred_annotations = {}
        for pred_ann in prediction_json_folder.glob('*.json'):
            with open(prediction_json_folder / pred_ann, 'r') as f:
                tmp_json = json.load(f)
            pred_annotations.update({el['image_id']: el for el in tmp_json['annotations']})

        matched_annotations_list = []
        for gt_ann in gt_json['annotations']:
            image_id = gt_ann['image_id']
            if image_id in pred_annotations:
                matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

        pq_stat = pq_compute_multi_core(matched_annotations_list, self._panoptic_gt_folder, prediction_path, categories)
        results = average_pq(pq_stat, categories)

        category_ids = sorted(results['per_class'].keys())
        for category_id in category_ids:
            results['per_class'][categories[category_id]['name']] = results['per_class'].pop(
                category_id)

        return results

    @property
    def pipeline(self) -> List[dict]:
        """
        Getter of the data loading pipeline.

        :return: data loading pipeline
        :rtype: list
        """
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value):
        """
        Setter for the data loading pipeline

        :param value: data loading pipeline
        :type value: list
        """
        self._pipeline = value

    def _build_mmdet_dataset(self,
                             test_mode: bool = False
                             ) -> MmdetCityscapesDataset:
        """
        Generates the mmdet representation of the dataset to be used with the mmdet API.

        :param test_mode: if set to True, the panoptic ground truth data has to be present
        :type test_mode: bool
        """
        if self.pipeline is None:
            raise ValueError('No dataset pipeline has been set.')

        config = {
            'ann_file': str(self._annotation_file),
            'img_prefix': str(self._image_folder),
            'seg_prefix': str(self._segmentation_folder),
            'type': 'CityscapesDataset',
            'pipeline': self.pipeline
        }
        if test_mode:
            if not self._panoptic_gt_folder.exists():
                raise RuntimeError('Dataset does not contain panoptic ground truth values.')
            config['panoptic_gt'] = str(self._panoptic_gt_folder)
            mmdet_dataset = build_dataset(config, {'test_mode': True})
        else:
            mmdet_dataset = build_dataset(config)
        return mmdet_dataset

    def __getitem__(self, idx: int) -> Tuple[Image, None]:
        """
        This method is used for loading the idx-th sample of a dataset along with its annotation.
        In this case, the annotation is split up into different files and, thus, a different interface is used.

        :param idx: the index of the sample to load
        :type idx: int
        :return: the idx-th sample and a placeholder for the corresponding annotation
        :rtype: Tuple of (Image, None)
        """
        image_filename = self._image_filenames[idx]
        image = Image.open(str(image_filename))

        return image, None

    def __len__(self) -> int:
        """
        This method returns the size of the dataset.

        :return: the size of the dataset
        :rtype: int
        """
        return len(self._image_filenames)

    @staticmethod
    def prepare_data(input_path: Union[str, bytes, os.PathLike], output_path: Union[str, bytes, os.PathLike],
                     generate_train_evaluation: bool = False, num_workers: int = mp.cpu_count()):
        """
        Convert the raw Cityscapes dataset to match the expected folder structure.

        :param input_path: path to the raw Cityscapes dataset
        :type input_path: str, bytes, PathLike
        :param output_path: path to the converted Cityscapes dataset
        :type output_path: str, bytes, PathLike
        :param generate_train_evaluation: if set to True, the training set will prepared to be used for evaluation.
            Usually, this is not required.
        :type generate_train_evaluation: bool
        :param num_workers: number of workers to be used in parallel
        :type num_workers: int
        """
        if not isinstance(input_path, Path):
            input_path = Path(input_path)
        if not isinstance(output_path, Path):
            output_path = Path(output_path)

        splits = {
            "train": ("leftImg8bit/train", "gtFine/train"),
            "val": ("leftImg8bit/val", "gtFine/val"),
        }

        if not input_path.exists():
            raise ValueError(f'The specified input path does not exist: {input_path}')
        if output_path.exists():
            raise ValueError('The specified output path already exists.')
        if not (input_path / 'leftImg8bit').exists():
            raise ValueError('Please download and extract the image files first: leftImg8bit_trainvaltest.zip')
        if not (input_path / 'gtFine').exists():
            raise ValueError(
                'Please download and extract the ground truth fine annotations first: gtFine_trainvaltest.zip')

        # COCO-style category list
        coco_categories = []
        for label in cs_labels:
            if label.trainId != 255 and label.trainId != -1 and label.hasInstances:
                coco_categories.append({"id": label.trainId, "name": label.name})

        coco_out = {
            "info": {"version": "1.0"},
            "images": [],
            "categories": coco_categories,
            "annotations": []
        }

        # Process splits
        for split, (split_img_subdir, split_mask_subdir) in splits.items():
            img_split_dir = output_path / split / 'images'
            mask_split_dir = output_path / split / 'stuffthingmaps'
            img_split_dir.mkdir(parents=True)
            mask_split_dir.mkdir(parents=True)

            img_input_dir = input_path / split_img_subdir
            mask_input_dir = input_path / split_mask_subdir
            img_list = [(file.parent.name, file.stem.replace('_gtFine_instanceIds', ''), 'gtFine') for file in
                        mask_input_dir.glob('*/*_instanceIds.png')]

            # Convert to COCO detection format
            with tqdm(total=len(img_list), desc=f'Converting {split}') as pbar:
                with mp.Pool(processes=num_workers, initializer=_Counter.init_counter, initargs=(_Counter(0),)) as pool:
                    for coco_img, coco_ann in pool.imap(
                            partial(
                                _process_data,
                                image_input_dir=img_input_dir,
                                mask_input_dir=mask_input_dir,
                                image_output_dir=img_split_dir,
                                mask_output_dir=mask_split_dir
                            ),
                            img_list
                    ):
                        coco_out["images"].append(coco_img)
                        coco_out["annotations"] += coco_ann
                        pbar.update(1)

            # Write COCO detection format annotation
            with open(output_path / split / 'annotations.json', "w") as f:
                json.dump(coco_out, f, indent=4)

        # Generate panoptic ground truth data used during evaluation
        set_names = ['val']
        if generate_train_evaluation:
            set_names.append('train')
        for split in set_names:
            convert2panoptic(
                cityscapesPath=input_path / 'gtFine',
                outputFolder=output_path,
                setNames=[split]
            )
            shutil.move(output_path / f'cityscapes_panoptic_{split}', output_path / split / 'panoptic_gt')
            shutil.move(output_path / f'cityscapes_panoptic_{split}.json', output_path / split / 'panoptic_gt.json')


def _process_data(image, image_input_dir: Path, mask_input_dir: Path, image_output_dir: Path, mask_output_dir: Path):
    img_dir, img_id, lbl_cat = image
    img_unique_id = counter.increment()
    coco_ann = []

    # Load the annotation
    with PilImage.open(mask_input_dir / img_dir / f"{img_id}_{lbl_cat}_instanceIds.png") as lbl_img:
        lbl = np.array(lbl_img)
        lbl_size = lbl_img.size
    ids = np.unique(lbl)

    # Compress the labels and compute cat
    lbl_out = np.ones(lbl.shape, np.uint8) * 255
    for city_id in ids:
        if city_id < 1000:
            # Stuff or group
            cls_i = city_id
            iscrowd_i = cs_labels[cls_i].hasInstances
        else:
            # Instance
            cls_i = city_id // 1000
            iscrowd_i = False

        # If it's a void class just skip it
        if cs_labels[cls_i].trainId == 255 or cs_labels[cls_i].trainId == -1:
            continue

        # Extract all necessary information
        iss_class_id = cs_labels[cls_i].trainId

        mask_i = lbl == city_id

        lbl_out[mask_i] = iss_class_id

        # Compute COCO detection format annotation
        if cs_labels[cls_i].hasInstances:
            category_info = {"id": iss_class_id, "is_crowd": iscrowd_i}
            coco_ann_i = pct.create_annotation_info(counter.increment(), img_unique_id, category_info, mask_i, lbl_size)
            if coco_ann_i is not None:
                coco_ann.append(coco_ann_i)

        # COCO detection format image annotation
        coco_img = pct.create_image_info(img_unique_id, f'{img_id}.png', lbl_size)

        # Write output
        PilImage.fromarray(lbl_out).save(mask_output_dir / f'{img_id}.png')
        shutil.copy(image_input_dir / img_dir / f'{img_id}_leftImg8bit.png', image_output_dir / f'{img_id}.png')

        return coco_img, coco_ann


class _Counter:
    def __init__(self, initval: int = 0):
        self._value = mp.Value('i', initval)

    def increment(self, n: int = 1):
        with mp.Lock():
            value = self._value.value
            self._value.value += n
        return value

    @staticmethod
    def init_counter(c):
        global counter
        counter = c
