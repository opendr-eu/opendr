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

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import pickle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from torch import Tensor
from numpy.typing import ArrayLike

from opendr.engine.data import Image
from opendr.engine.learners import Learner

from opendr.perception.continual_slam.algorithm.depth_pose_module import DepthPoseModule
from opendr.perception.continual_slam.configs.config_parser import ConfigParser


class ContinualSLAMLearner(Learner):
    """
    This class implements the CL-SLAM algorithm, which is a continual learning approach for SLAM.
    """
    def __init__(self,
                 config_file: Path,
                 mode: str = 'predictor',
                 ros: bool = False
                 ):
        """
        :param config_file: path to the config file
        :type config_file: Path
        :param mode: mode of the learner, either predictor or learner. Either 'predictor' or 'learner'
        :type mode: str
        :param ros: whether the learner is used in ROS environment
        :type ros: bool
        """
        self.config_file = ConfigParser(config_file)
        self.dataset_config = self.config_file.dataset
        self.height = self.dataset_config.height
        self.width = self.dataset_config.width
        self.model_config = self.config_file.depth_pose
        self.is_ros = ros
        super(ContinualSLAMLearner, self).__init__(lr=self.model_config.learning_rate)

        self.mode = mode

        if self.mode == 'predictor':
            # Create the predictor object
            self.predictor = DepthPoseModule(self.model_config, self.dataset_config, use_online=False, mode=mode)
            self.predictor.load_model(load_optimizer=True)
        elif self.mode == 'learner':
            self.learner = DepthPoseModule(self.model_config, self.dataset_config, use_online=False, mode=mode)
            self.learner.load_model(load_optimizer=True)
            depth_pose_config = self.config_file.depth_pose
            for g in self.learner.optimizer.param_groups:
                g['lr'] = depth_pose_config.learning_rate
        else:
            raise ValueError('Mode should be either predictor or learner')

    def fit(self,
            batch: Tuple[Dict, None],
            return_losses: bool = False,
            ) -> Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]:
        """
        In the context of CL-SLAM, we implemented fit method as adapt method, which updates the weights of learner
        based on coming input. Only works in learner mode.
        :param batch: tuple of (input, target)
        :type batch: Tuple[Dict, None]
        :param return_losses: whether to return the losses
        :type return_losses: bool
        :return: tuple of (prediction, loss)
        :rtype: Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]
        """
        if self.mode == 'learner':
            return self._fit(batch, return_losses)
        else:
            raise ValueError('Fit is only available in learner mode')

    def infer(self,
              batch: Tuple[Dict, None],
              return_losses: bool = False,
              ) -> Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]:
        """
        We implemented infer method as predict method which predicts the output based on coming input. Only works in
        predictor mode.
        :param batch: tuple of (input, target)
        :type batch: Tuple[Dict, None]
        :param return_losses: whether to return the losses
        :type return_losses: bool
        :return: tuple of (prediction, loss)
        :rtype: Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]
        """
        if self.mode == 'predictor':
            return self._predict(batch, return_losses)
        else:
            raise ValueError('Inference is only available in predictor mode')

    def save(self) -> str:
        """
        Save the model weights as an binary-encoded string for ros message
        """
        save_dict = {}
        location = Path.cwd() / 'temp_weights'
        if not location.exists():
            location.mkdir(parents=True, exist_ok=True)
        for model_name, model in self.learner.models.items():
            if model is None:
                continue
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            to_save = model.state_dict()
            if 'encoder' in model_name:
                # ToDo: look into this
                # Save the sizes - these are needed at prediction time
                to_save['height'] = Tensor(self.height)
                to_save['width'] = Tensor(self.width)
            if not self.is_ros:
                save_path = location / f'{model_name}.pth'
                torch.save(to_save, save_path)

        if self.is_ros:
            save_dict = pickle.dumps(save_dict).decode('latin1')
            return save_dict
        else:
            return str(location)

    def load(self, weights_folder: str = None, message: str = None, load_optimizer: bool = False) -> None:
        """
        Load the model weights from an binary-encoded string for ros message
        """
        if self.is_ros:
            load_dict = pickle.loads(bytes(message, encoding='latin1'))
            for model_name, model in self.predictor.models.items():
                if model is None:
                    continue
                if isinstance(model, torch.nn.DataParallel):
                    model = model.module
                model.load_state_dict(load_dict[model_name])
        else:
            self.predictor.load_model(weights_folder = weights_folder, load_optimizer=load_optimizer)

    def eval(self, dataset, *args, **kwargs):
        raise NotImplementedError

    def optimize(self, target_device):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    # Private methods ================================================================================
    def _fit(self,
             batch: Tuple[Dict, None],
             return_losses: bool = False,
             ) -> Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]:
        """
        :param batch: tuple of (input, target)
        :type batch: Tuple[Dict, None]
        :param return_losses: whether to return the losses
        :type return_losses: bool
        :return: tuple of (prediction, loss)
        :rtype: Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]
        """
        input_dict = self._input_formatter(batch)
        # Adapt
        if return_losses:
            prediction, losses = self.learner.adapt(input_dict, steps=5, return_loss=return_losses)
            return self._output_formatter(prediction), losses
        else:
            prediction = self.learner.adapt(input_dict, steps=5, return_loss=return_losses)
            return self._output_formatter(prediction), None

    def _predict(self,
                 batch: Tuple[Dict, None],
                 return_losses: bool = False,
                 ) -> Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]:
        """
        :param batch: tuple of (input, target)
        :type batch: Tuple[Dict, None]
        :return: tuple of (prediction, loss)
        :rtype: Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]
        """
        input_dict = self._input_formatter(batch)
        # Get the prediction
        if return_losses:
            prediction, losses = self.predictor.predict(input_dict, return_loss=return_losses)
            # Convert the prediction to opendr format
            return self._output_formatter(prediction), losses
        else:
            prediction = self.predictor.predict(input_dict, return_loss=return_losses)
            # Convert the prediction to opendr format
            return self._output_formatter(prediction)

    def _input_formatter(self, batch: Tuple[Dict, None]) -> Dict[Any, Tensor]:
        """
        Format the input for the prediction
        :param batch: tuple of (input, target)
        :type batch: Tuple[Dict, None]
        :return: dictionary of input
        :rtype: Dict[Any, Tensor]
        """
        if type(batch) == dict:
            inputs = batch
        else:
            inputs = batch[0]

        # Create a dictionary with frame ids as [-1, 0, 1]
        input_dict = {}
        for frame_id, id in zip([-1, 0, 1], inputs.keys()):
            input_dict[(frame_id, 'image')] = torch.Tensor(inputs[id][0].data)
            input_dict[(frame_id, 'distance')] = torch.Tensor([inputs[id][1]])
        return input_dict

    def _output_formatter(self, prediction: Dict) -> Tuple[ArrayLike, ArrayLike]:
        """
        Format the output of the prediction
        :param prediction: dictionary of predictions which has items of:
        frame_id -> (depth, pose)
        depth -> Tensors of shape (1, 1, H, W). The number of tensors is equal to scales
        pose -> 6 Tensors, which we will only use cam_T_cam for 0->1 since it is the odometry
        """
        # Convert the prediction to opendr format
        for item in prediction:
            if item[0] == 'disp':
                if item[2] == 0:
                    depth = self._colorize_depth(prediction[item].squeeze().cpu().detach().numpy())
            if item[0] == 'cam_T_cam':
                if item[2] == 1:
                    odometry = prediction[item].cpu().detach().numpy()
        return depth, odometry

    def _colorize_depth(self, depth):
        import cv2
        vmax = np.percentile(depth, 95)
        normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
        mapper = plt.cm.ScalarMappable(norm=normalizer, cmap="magma")
        colormapped_img = (mapper.to_rgba(depth.squeeze())[:, :, :3]*255).astype(np.uint8)
        colormapped_img = cv2.cvtColor(colormapped_img, cv2.COLOR_RGB2BGR)
        return Image(colormapped_img)

    # ================================================================================================

# TODO: Delete this later since it is just for debugging and testing
if __name__ == "__main__":
    local_path = Path(__file__).parent / 'configs'
    learner = ContinualSLAMLearner(local_path / 'singlegpu_kitti.yaml', mode='learner', ros=False)

    # Test the learner/predictor
    from opendr.perception.continual_slam.datasets.kitti import KittiDataset
    dataset_config = ConfigParser(local_path / 'singlegpu_kitti.yaml').dataset
    dataset_path = dataset_config.dataset_path
    dataset = KittiDataset(str(dataset_path), dataset_config)

    from PIL import Image as imgg 
    import time

    for i, batch in enumerate(dataset):
        if i < 5:
            continue
        depth, odometry = learner.fit(batch)
        # depth, odometry = predictor.infer(batch)
        message = learner.save()
        # predictor.load(message)