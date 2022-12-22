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



from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.data import Image
from opendr.engine.learners import Learner
from opendr.engine.target import Heatmap


from opendr.perception.continual_slam.algorithm.depth_pose_prediction.networks import ResnetEncoder, DepthDecoder, PoseDecoder
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.config import Config


class ContinualSLAMLearner(Learner):
    def __init__(self,
                 config_file: Config,
                 ):
        """
        This is the ContinualSLAMLearner class, which implements the continual SLAM algorithm.
        :param config_file: Path to the config file.
        :param dataset_config: Path to the dataset config file.
        :param checkpoint_file: Path to the checkpoint file.
        """
        super(ContinualSLAMLearner, self).__init__(lr=config_file.learning_rate)

        self.models = {}
        self.models['depth_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained)
        self.models['depth_decoder'] = DepthDecoder(self.models['depth_encoder'].num_ch_encoder,
                                                    self.scales)
        self.models['pose_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained,
                                                    self.num_pose_frames)
        self.models['pose_decoder'] = PoseDecoder(self.models['pose_encoder'].num_ch_encoder,
                                                  num_input_features=1,
                                                  num_frames_to_predict_for=2)

