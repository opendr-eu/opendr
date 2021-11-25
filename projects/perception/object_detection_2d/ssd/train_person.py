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
from opendr.perception.object_detection_2d.datasets.detection_dataset import ConcatDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", help="Dataset to train on", type=str, default="voc", choices=["voc", "coco",
    #                                                                                                "widerperson"])
    # parser.add_argument("--data-root", help="Dataset root folder", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=6)
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=1e-5)
    parser.add_argument("--val-after", help="Epochs in-between  evaluations", type=int, default=1)
    parser.add_argument("--checkpoint-freq", help="Frequency in-between checkpoint saving", type=int, default=5)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=50)
    parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from", type=int, default=0)

    args = parser.parse_args()

    train_dataset_1 = WiderPersonDataset(root='/home/administrator/data/wider_person', splits=['train'])
    val_dataset = XMLBasedDataset(root='/home/administrator/data/agi_human_data', dataset_type='agi_human',
                                 images_dir='human', annotations_dir='human_anot', classes=['person'])
    train_dataset_2 = XMLBasedDataset(root='/home/administrator/data/agi_human_data', dataset_type='agi_human',
                                           images_dir='no_human', annotations_dir='no_human_anot', classes=['person'])
    # train_dataset_3 = XMLBasedDataset(root='/home/administrator/data/agi_human_data', dataset_type='agi_human',
    #                                     images_dir='human', annotations_dir='human_anot', classes=['person'])
    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    # train_dataset = train_dataset_1

    ssd = SingleShotDetectorLearner(device=args.device, batch_size=args.batch_size, lr=args.lr, val_after=args.val_after,
                                    checkpoint_load_iter=args.resume_from, epochs=args.n_epochs,
                                    checkpoint_after_iter=args.checkpoint_freq, log_after=50)

    ssd.fit(train_dataset, val_dataset)
    ssd.save("./ssd_concat_person_model")
