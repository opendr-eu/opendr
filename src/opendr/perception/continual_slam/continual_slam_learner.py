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


from opendr.perception.continual_slam.algorithm.datasets import Config as DatasetConfig
from opendr.perception.continual_slam.algorithm.depth_pose_prediction import DepthPosePrediction
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.config import DepthPosePrediction as Config

class ContinualSLAMLearner(Learner):
    def __init__(self,
                 config_file: Config,
                 dataset_config: DatasetConfig,
                 checkpoint_file = None,
                 type : str = 'publisher',
                 ):
        """
        This is the ContinualSLAMLearner class, which implements the continual SLAM algorithm.
        :param config_file: Path to the config file.
        :param dataset_config: Path to the dataset config file.
        :param checkpoint_file: Path to the checkpoint file.
        """
        super(ContinualSLAMLearner, self).__init__(lr=config_file.learning_rate)
        self.config_file = config_file
        self.dataset_config = dataset_config
        self.checkpoint_file = checkpoint_file
        
        if type == 'publisher':
            self.type = False
        elif type == 'learner':
            self.type = True
        else:
            raise ValueError('type should be either publisher or learner')
        
        if self.type:
            self.model = DepthPosePrediction(config_file, dataset_config, use_online=True)
            self.model.load_model(load_optimizer=True)
            self.model.load_online_model(load_optimizer=True)
        else: 
            self.model = DepthPosePrediction(config_file, dataset_config, use_online=False)
            self.model.load_model(load_optimizer=True)
        
    def inference(self):
        pass

    def adapt(self, inputs):
        if not self.type:
            raise ValueError('adapt() is only available for learner')
        
        self.model.adapt(inputs)

    def update(self):
        pass
    

