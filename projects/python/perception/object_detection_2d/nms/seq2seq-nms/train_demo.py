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


from opendr.perception.object_detection_2d import Seq2SeqNMSLearner
import os
import argparse
OPENDR_HOME = os.environ['OPENDR_HOME']

parser = argparse.ArgumentParser()
parser.add_argument("--app_feats", help="Type of appearance-based features", type=str, default="fmod",
                    choices=["fmod", "zeros"])
parser.add_argument("--fmod_type", help="Type of fmod maps", type=str, default="EDGEMAP",
                    choices=["EDGEMAP", "FAST", "AKAZE", "BRISK", "ORB"])
parser.add_argument("--iou_filtering", help="Pre-processing IoU threshold", type=float, default=1.0)
parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=1e-4)
parser.add_argument("--n_epochs", help="Number of total epochs", type=int, default=10)
parser.add_argument("--tmp_path", help="Temporary path where weights will be saved", type=str,
                    default=os.path.join(OPENDR_HOME, 'projects/python/perception/object_detection_2d/nms/seq2seq-nms/tmp'))
parser.add_argument("--checkpoint_freq", help="Frequency in-between checkpoint saving", type=int, default=1)
parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from", type=int, default=0)
parser.add_argument("--dataset", help="Dataset to train on", type=str, default="PETS", choices=["PETS", "COCO",
                                                                                                "TEST_MODULE"])
parser.add_argument("--use_ssd", help="Train using SSD as default detector", type=bool, default=False)
parser.add_argument("--max_dt_boxes", help="Maximum number of input RoIs fed to Seq2Seq-NMS", type=int, default=500)
parser.add_argument("--data-root", help="Dataset root folder", type=str,
                    default=os.path.join(OPENDR_HOME,
                                         'projects/python/perception/object_detection_2d/nms/seq2seq-nms/datasets'))
args = parser.parse_args()
seq2SeqNMSLearner = Seq2SeqNMSLearner(epochs=args.n_epochs, lr=args.lr, device=args.device, app_feats=args.app_feats,
                                      fmod_map_type=args.fmod_type, iou_filtering=args.iou_filtering,
                                      temp_path=args.tmp_path, checkpoint_after_iter=args.checkpoint_freq,
                                      checkpoint_load_iter=args.resume_from)
seq2SeqNMSLearner.fit(dataset=args.dataset, use_ssd=args.use_ssd,
                      datasets_folder=args.data_root, silent=False, verbose=True,
                      max_dt_boxes=args.max_dt_boxes)
seq2SeqNMSLearner.save(path=os.path.join(args.tmp_path, 'saved_model'), current_epoch=args.n_epochs-1,
                       max_dt_boxes=args.max_dt_boxes)
