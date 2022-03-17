# Copyright 2020-2022 OpenDR European Project
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


from opendr.perception.object_detection_2d.nms.seq2seq_nms.seq2seq_nms_learner import Seq2SeqNMSLearner
import os
OPENDR_HOME = os.environ['OPENDR_HOME']

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to train on", type=str, default="voc", choices=["PETS", "COCO",
                                                                                                   "TEST_MODULE"])
    parser.add_argument("--data_root", help="Dataset root folder", type=str)
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_dt_boxes", help="Maximum number of input RoIs to Seq2Seq-NMS", type=int, default=450)
    parser.add_argument("--nms_gt_iou", help="Learning rate to use for training", type=float, default=5e-4)
    parser.add_argument("--val-after", help="Epochs in-between  evaluations", type=int, default=5)
    parser.add_argument("--checkpoint-freq", help="Frequency in-between checkpoint saving", type=int, default=5)
    parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=25)
    parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from", type=int, default=0)

    args = parser.parse_args()

seq2seq_tmp_path = OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/coco_tmp'
seq2seq_logs_path = OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/coco_logs'

seq2SeqNMSLearner = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering=None, app_feats='fmod',
                                      checkpoint_after_iter=1, temp_path=seq2seq_tmp_path, epochs=8)
seq2SeqNMSLearner.fit(dataset=args.dataset, use_ssd=False,
                      datasets_folder= args.data_root,
                      logging_path=seq2seq_logs_path, silent=False, verbose=True, nms_gt_iou=0.50,
                      max_dt_boxes=args.max_dt_boxes)
