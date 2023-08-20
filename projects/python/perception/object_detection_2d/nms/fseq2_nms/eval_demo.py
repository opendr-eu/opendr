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
parser.add_argument("--iou_filtering", help="Pre-processing IoU threshold", type=float, default=0.8)
parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--pretrained_model", help="Name of pretrained model", type=str, default='fseq2_pets_ssd_pets',
                    choices=['fseq2_pets_ssd'])
parser.add_argument("--split", help="The split of the corresponding dataset", type=str, default='test',
                    choices=["test", "val", "train"])
parser.add_argument("--max_dt_boxes", help="Maximum number of input RoIs fed to FSeq2-NMS", type=int, default=800)
parser.add_argument("--dataset", help="Dataset to train on", type=str, default="PETS", choices=["PETS", "COCO",
                                                                                                "TEST_MODULE"])
parser.add_argument("--ssd_model", help="SSD model used for feeding RoIS to the NMS procedure", type=str,
                    default='ssd_512_vgg16_atrous_pets', choices=['ssd_512_vgg16_atrous_pets', 'ssd_default_person'])
parser.add_argument("--data_root", help="Dataset root folder", type=str,
                    default=os.path.join(OPENDR_HOME,
                                         'projects/python/perception/object_detection_2d/nms/datasets'))
parser.add_argument("--post_thres", help="Confidence threshold, used for RoI selection after FSeq2-NMS rescoring",
                    type=float, default=0.0)
parser.add_argument("--tmp_path", help="Temporary path for saving output data", type=str,
                    default=os.path.join(OPENDR_HOME,
                                         'projects/python/perception/object_detection_2d/nms/fseq2_nms/tmp_pets'))
args = parser.parse_args()
fseqNMSLearner = FSeq2NMSLearner(device=args.device, iou_filtering=args.iou_filtering, temp_path=args.tmp_path,
                                 app_input_dim=315)
#fseqNMSLearner.download(model_name=args.pretrained_model, path=args.tmp_path)
fseqNMSLearner.load('/media/fastdata/charsyme/opendr_sum23/opendr/projects/python/perception/object_detection_2d/nms/fseq2_nms/tmp_pets_pets5/checkpoint_epoch_7.json', verbose=True)
#fseqNMSLearner.load(os.path.join(args.tmp_path, args.pretrained_model), verbose=True)
fseqNMSLearner.eval(dataset=args.dataset, split=args.split, max_dt_boxes=args.max_dt_boxes,
                    datasets_folder=args.data_root, threshold=args.post_thres, ssd_model=args.ssd_model)
