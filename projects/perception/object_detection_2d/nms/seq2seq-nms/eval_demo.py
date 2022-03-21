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

from opendr.perception.object_detection_2d.nms import Seq2SeqNMSLearner
import os
OPENDR_HOME = os.environ['OPENDR_HOME']
temp_path = OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/pets_tmp'
dataset_path = OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/pets_dataset'
seq2SeqNMSLearner = Seq2SeqNMSLearner(iou_filtering=0.8, app_feats='fmod', temp_path=temp_path,
                                      device='cuda')
seq2SeqNMSLearner.download(model_name='seq2seq_pets_jpd', path=temp_path)
seq2SeqNMSLearner.load(os.path.join(temp_path, 'seq2seq_pets_jpd'), verbose=True)
seq2SeqNMSLearner.eval(dataset='PETS', split='test', max_dt_boxes=600,
                       datasets_folder=dataset_path, use_ssd=False)
