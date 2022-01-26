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

import argparse

from opendr.engine.datasets import ExternalDataset
from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner
from opendr.perception.object_detection_2d.datasets.wider_person import WiderPersonDataset
from opendr.perception.object_detection_2d.datasets.xmldataset import XMLBasedDataset
from opendr.perception.object_detection_2d.datasets.jsondataset import JSONBasedDataset
from opendr.perception.object_detection_2d.datasets.detection_dataset import ConcatDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to train on", type=str, default="coco", choices=["voc", "coco",
                                                                                                   "widerperson"])
    #parser.add_argument("--data-root", help="Dataset root folder", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=2)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=5e-4)
    parser.add_argument("--val-after", help="Epochs in-between  evaluations", type=int, default=1)
    parser.add_argument("--checkpoint-freq", help="Frequency in-between checkpoint saving", type=int, default=1)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=30)
    parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from", type=int, default=0)

    args = parser.parse_args()

    listCocoClasses = ["person", 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                       'dog', 'horse', 'sheep', 'cow', 'elephant','bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                       'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                       'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                       'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                       'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

    data_root = '/home/manos/data/coco2017/'

    #train_dataset = WiderPersonDataset(root='/home/manos/data/wider_person', splits=['train'])

    #dataset = ExternalDataset(data_root, 'coco')
    #val_dataset = ExternalDataset(data_root, 'coco')
    #val_dataset = JSONBasedDataset(root='/home/manos/data/coco2017/', dataset_type='coco',
    #                                images_dir='val2017', annotations_dir='annotations',
    #                                json_file='instances_val2017.json')#, classes=['person'])

    train_dataset_1 = XMLBasedDataset(root='/home/manos/data/agi_human_data/dataset_final/train', dataset_type='agi_human',
                                    images_dir='human', annotations_dir='human_anot', classes=['person'])
    train_dataset_2 = XMLBasedDataset(root='/home/manos/data/agi_human_data/dataset_final/train', dataset_type='agi_human_temp',
                                      images_dir='no_human', annotations_dir='no_human_anot', classes=['person'])

    val_dataset_1 = XMLBasedDataset(root='/home/manos/data/agi_human_data/dataset_final/test', dataset_type='agi_human',
                                  images_dir='human', annotations_dir='human_anot', classes=['person'])
    val_dataset_2 = XMLBasedDataset(root='/home/manos/data/agi_human_data/dataset_final/test', dataset_type='agi_human',
                                    images_dir='no_human', annotations_dir='no_human_anot', classes=['person'])

    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    val_dataset = ConcatDataset([val_dataset_1, val_dataset_2])

    ssd = SingleShotDetectorLearner(device=args.device, batch_size=args.batch_size, lr=args.lr, val_after=args.val_after,
                                    checkpoint_load_iter=args.resume_from, epochs=args.n_epochs,
                                    checkpoint_after_iter=args.checkpoint_freq,lr_decay=0.0001, lr_decay_epoch=[2, 4, 6],
                                    lr_schedule='', fine_tuning=False)

    ssd.load_gcv('coco', keep_classes=['person'])
    #ssd.load_gcv('coco', keep_classes=listCocoClasses)
    ssd.fit(train_dataset, val_dataset, 'TensorboardLogs/Training_agi_temp')
    ssd.save("./ssd_agi_temp")
