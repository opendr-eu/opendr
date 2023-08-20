# Copyright 2020-2023 OpenDR European Project
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


from opendr.perception.object_detection_2d import FSeq2NMSLearner
import os
import argparse
OPENDR_HOME = os.environ['OPENDR_HOME']

parser = argparse.ArgumentParser()
parser.add_argument("--iou_filtering", help="Pre-processing IoU threshold", type=float, default=1.0)
parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=1e-4)
parser.add_argument("--n_epochs", help="Number of total epochs", type=int, default=14)
parser.add_argument("--tmp_path", help="Temporary path where weights will be saved", type=str,
                    default=os.path.join(OPENDR_HOME, 'projects/python/perception/object_detection_2d/nms/fseq2-nms/tmp'))
parser.add_argument("--checkpoint_freq", help="Frequency in-between checkpoint saving", type=int, default=1)
parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from", type=int, default=0)
parser.add_argument("--dataset", help="Dataset to train on", type=str, default="PETS", choices=["PETS", "CROWDHUMAN",
                                                                                                "TEST_MODULE"])
parser.add_argument("--ssd_model", help="SSD model used for feeding RoIS to the NMS procedure", type=str,
                    default='ssd_512_vgg16_atrous_pets', choices=['ssd_512_vgg16_atrous_pets', 'ssd_default_person'])
parser.add_argument("--max_dt_boxes", help="Maximum number of input RoIs fed to Seq2Seq-NMS", type=int, default=500)
parser.add_argument("--data-root", help="Dataset root folder", type=str,
                    default=os.path.join(OPENDR_HOME,
                                         'projects/python/perception/object_detection_2d/nms/datasets'))
args = parser.parse_args()
fseqNMSLearner = FSeq2NMSLearner(epochs=args.n_epochs, lr=args.lr, device=args.device, iou_filtering=args.iou_filtering,
                                 temp_path=args.tmp_path, checkpoint_after_iter=args.checkpoint_freq,
                                 checkpoint_load_iter=args.resume_from)
# fseqNMSLearner.load(os.path.join(args.tmp_path, 'checkpoint_epoch_4', 'checkpoint_epoch_4.json'), verbose=True)
fseqNMSLearner.fit(dataset=args.dataset, datasets_folder=args.data_root, ssd_model=args.ssd_model, silent=False,
                   verbose=True, max_dt_boxes=args.max_dt_boxes)
fseqNMSLearner.save(path=os.path.join(args.tmp_path, 'saved_model'), current_epoch=args.n_epochs-1,
                    max_dt_boxes=args.max_dt_boxes)
