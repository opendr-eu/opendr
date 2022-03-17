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

seq2seq_tmp_path = OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/coco_tmp'
seq2SeqNMSLearner = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering=None, app_feats='fmod',
                                      checkpoint_after_iter=1, temp_path=seq2seq_tmp_path, epochs=8)
seq2SeqNMSLearner.fit(dataset='COCO', use_ssd=False,
                      datasets_folder=OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/datasets',
                      logging_path='./logs_coco_own3', silent=False, verbose=True, nms_gt_iou=0.50,
                      max_dt_boxes=450)
