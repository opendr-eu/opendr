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
parser.add_argument("--pretrained_model", help="Name of pretrained model", type=str, default='seq2seq_pets_jpd_fmod',
                    choices=['seq2seq_pets_jpd'])
parser.add_argument("--split", help="The split of the corresponding dataset", type=str, default='test',
                    choices=["test", "val", "train"])
parser.add_argument("--max_dt_boxes", help="Maximum number of input RoIs fed to Seq2Seq-NMS", type=int, default=600)
parser.add_argument("--dataset", help="Dataset to train on", type=str, default="PETS", choices=["PETS", "COCO",
                                                                                                "TEST_MODULE"])
parser.add_argument("--data_root", help="Dataset root folder", type=str,
                    default=os.path.join(OPENDR_HOME,
                                         'projects/python/perception/object_detection_2d/nms/seq2seq-nms/datasets'))
parser.add_argument("--use_ssd", help="Train using SSD as detector", type=bool, default=False)
parser.add_argument("--post_thres", help="Confidence threshold, used for RoI selection after seq2seq-nms rescoring",
                    type=float, default=0.0)

args = parser.parse_args()
tmp_path = os.path.join(OPENDR_HOME, 'projects/python/perception/object_detection_2d/nms/seq2seq-nms/tmp')
seq2SeqNMSLearner = Seq2SeqNMSLearner(device=args.device, app_feats=args.app_feats, fmod_map_type=args.fmod_type,
                                      iou_filtering=args.iou_filtering,
                                      temp_path=tmp_path)
seq2SeqNMSLearner.download(model_name=args.pretrained_model, path=tmp_path)
seq2SeqNMSLearner.load(os.path.join(tmp_path, args.pretrained_model), verbose=True)
seq2SeqNMSLearner.eval(dataset=args.dataset, use_ssd=args.use_ssd, split=args.split, max_dt_boxes=args.max_dt_boxes,
                       datasets_folder=args.data_root, threshold=args.post_thres)
