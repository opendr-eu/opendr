# Copyright 2020-2024 OpenDR European Project
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
from opendr.perception.object_detection_2d import SingleShotDetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.engine.data import Image
import os
import argparse
OPENDR_HOME = os.environ['OPENDR_HOME']

parser = argparse.ArgumentParser()
parser.add_argument("--app_feats", help="Type of appearance-based features", type=str, default="fmod",
                    choices=["fmod", "zeros"])
parser.add_argument("--fmod_type", help="Type of fmod maps", type=str, default="EDGEMAP",
                    choices=["EDGEMAP", "FAST", "AKAZE", "BRISK", "ORB"])
parser.add_argument("--iou_filtering", help="Pre-processing IoU threshold", type=float, default=1.0)
parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda",
                    choices=["cuda", "cpu"])
parser.add_argument("--pretrained_model", help="Name of pretrained model", type=str,
                    default='seq2seq_pets_ssd_pets_fmod',
                    choices=['seq2seq_pets_jpd_pets_fmod', 'seq2seq_pets_ssd_wider_person_fmod',
                             'seq2seq_pets_ssd_pets_fmod', 'seq2seq_coco_frcn_coco_fmod',
                             'seq2seq_coco_ssd_wider_person_fmod'])
parser.add_argument("--ssd_model", help="SSD model used for feeding RoIS to the NMS procedure", type=str,
                    default='ssd_512_vgg16_atrous_pets', choices=['ssd_512_vgg16_atrous_pets', 'ssd_default_person'])
parser.add_argument("--tmp_path", help="Temporary path for saving output data", type=str,
                    default=os.path.join(OPENDR_HOME,
                                         'projects/python/perception/object_detection_2d/nms/seq2seq_nms/tmp'))
args = parser.parse_args()
seq2SeqNMSLearner = Seq2SeqNMSLearner(device=args.device, app_feats=args.app_feats, fmod_map_type=args.fmod_type,
                                      iou_filtering=args.iou_filtering,
                                      temp_path=args.tmp_path)
seq2SeqNMSLearner.download(model_name=args.pretrained_model, path=args.tmp_path)
seq2SeqNMSLearner.load(os.path.join(args.tmp_path, args.pretrained_model), verbose=True)

ssd = SingleShotDetectorLearner(device=args.device)
ssd.download(".", mode="pretrained")
ssd.load(args.ssd_model, verbose=True)
img = Image.open(OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/img_temp/frame_0000.jpg')
if not isinstance(img, Image):
    img = Image(img)
boxes = ssd.infer(img, threshold=0.3, custom_nms=seq2SeqNMSLearner)
draw_bounding_boxes(img.opencv(), boxes, class_names=ssd.classes, show=True)
