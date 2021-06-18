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

from typing import Tuple, Any, Dict, Union, List
from pathlib import Path
import mmcv
import json

from opendr.perception.panoptic_segmentation.datasets.utils import Image
from opendr.engine.datasets import ExternalDataset, DatasetIterator

from cityscapesscripts.evaluation.evalPanopticSemanticLabeling import pq_compute_multi_core, average_pq

from mmdet.datasets import build_dataset
from mmdet.datasets import CityscapesDataset as MmdetCityscapesDataset


class CityscapesDataset(ExternalDataset, DatasetIterator):
    """
    The CityscapesDataset class provides the OpenDR and mmdet APIs for different use cases. Inference using pre-trained
    models is supported by the OpenDR interface, for training and evaluation an instance of the respective mmdet version
    is created.

    The file structure should look like this:
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
        image = Image(mmcv.imread(image_filename), image_filename.name)

        return (image, None)

    def __len__(self) -> int:
        """
        This method returns the size of the dataset.

        :return: the size of the dataset
        :rtype: int
        """
        return len(self._image_filenames)
