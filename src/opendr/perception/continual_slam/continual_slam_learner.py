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

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

import torch
import cv2
import pickle
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from torch import Tensor
from numpy.typing import ArrayLike
from tqdm import tqdm
import urllib
from zipfile import ZipFile

from opendr.engine.data import Image
from opendr.engine.learners import Learner
from opendr.engine.constants import OPENDR_SERVER_URL

from opendr.perception.continual_slam.algorithm.depth_pose_module import DepthPoseModule
from opendr.perception.continual_slam.configs.config_parser import ConfigParser
from opendr.perception.continual_slam.algorithm.loop_closure.pose_graph_optimization import PoseGraphOptimization
from opendr.perception.continual_slam.algorithm.loop_closure.loop_closure_detection import LoopClosureDetection
from opendr.perception.continual_slam.algorithm.loop_closure.buffer import Buffer


class ContinualSLAMLearner(Learner):
    """
    This class implements the CL-SLAM algorithm, which is a continual learning approach for SLAM.
    """
    def __init__(self,
                 config_file: Union[str, Path],
                 mode: str = 'predictor',
                 ros: bool = False,
                 do_loop_closure: bool = False,
                 key_frame_freq: int = 5,
                 lc_distance_poses: int = 150,
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
        self.step = 0
        self.key_frame_freq = key_frame_freq
        self.do_loop_closure = do_loop_closure
        self.lc_distance_poses = lc_distance_poses
        self.since_last_lc = lc_distance_poses
        self.online_dataset = Buffer()
        self.pose_graph = PoseGraphOptimization()
        self.loop_closure_detection = LoopClosureDetection(self.config_file.loop_closure)
        super(ContinualSLAMLearner, self).__init__(lr=self.model_config.learning_rate)

        self.mode = mode

        if self.mode == 'predictor':
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
            batch: List,
            return_losses: bool = False,
            replay_buffer: bool = False,
            learner: bool = False):
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
            self._fit(batch, return_losses, replay_buffer, learner=learner)
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

    def save(self, location: str = None) -> str:
        """
        Save the model weights as an binary-encoded string for ros message
        """
        save_dict = {}
        if location is None:
            location = Path.cwd() / 'temp_weights'
        if not location.exists():
            location.mkdir(parents=True, exist_ok=True)
        for model_name, model in self.learner.models.items():
            if model is None:
                continue
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            to_save = model.state_dict()
            if 'encoder' in model_name and not self.is_ros:
                to_save['height'] = Tensor(self.height)
                to_save['width'] = Tensor(self.width)
            if not self.is_ros:
                save_path = location / f'{model_name}.pth'
                torch.save(to_save, save_path)
            if self.is_ros:
                save_dict[model_name] = to_save

        return str(location)

    def load(self, weights_folder: str = None, load_optimizer: bool = False) -> None:
        """
        Load the model weights from an binary-encoded string for ros message
        """
        self.predictor.load_model(weights_folder=weights_folder, load_optimizer=load_optimizer)

    def download(path: str, mode: str = 'model', trained_on: str = 'semantickitti', prepare_data: bool = False):
        """
        Download data from the OpenDR server.
        Valid modes include pre-trained model weights and data used in the unit tests.

        Currently, the following pre-trained models are available:
            - SemanticKITTI visual odometry

        :param path: path to the directory where the data should be downloaded
        :type path: str
        :param mode: mode to use. Valid options are ['model', 'test_data']
        :type mode: str
        :param trained_on: dataset on which the model was trained. Valid options are ['semantickitti']
        :type trained_on: str
        :param prepare_data: whether to prepare the data for the unit tests
        :type prepare_data: bool

        :return: path to the downloaded data
        :rtype: str
        """
        if mode == "model":
            models = {
                "semantickitti":
                f"{OPENDR_SERVER_URL}perception/continual_slam/models/model_semantickitti"
            }
            if trained_on not in models.keys():
                raise ValueError(f"Could not find model weights pre-trained on {trained_on}. "
                                 f"Valid options are {list(models.keys())}")
            url = models[trained_on]
        elif mode == "test_data":
            url = f"{OPENDR_SERVER_URL}perception/continual_slam/test_data.zip"
        else:
            raise ValueError("Invalid mode. Valid options are ['model', 'test_data']")

        networks = ["depth_encoder", "depth_decoder", "pose_encoder", "pose_decoder"]
        if not isinstance(path, Path):
            path = Path(path)
        filename = path
        path.mkdir(parents=True, exist_ok=True)

        def pbar_hook(prog_bar: tqdm):
            prev_b = [0]

            def update_to(b=1, bsize=1, total=None):
                if total is not None:
                    prog_bar.total = total
                prog_bar.update((b - prev_b[0]) * bsize)
                prev_b[0] = b

            return update_to

        if os.path.exists(filename) and os.path.isfile(filename):
            print(f'File already downloaded: {filename}')
        else:
            if mode == "model":
                for network in networks:
                    durl = url + f"/{network}.pth"
                    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1,
                              desc=f"Downloading {filename}") as pbar:
                        urllib.request.urlretrieve(durl, filename, pbar_hook(pbar))
            else:
                with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1,
                          desc=f"Downloading {filename}") as pbar:
                    urllib.request.urlretrieve(url, filename, pbar_hook(pbar))
        if prepare_data and mode == "test_data":
            print(f"Extracting {filename}")
            try:
                with ZipFile(filename, 'r') as zipObj:
                    zipObj.extractall(path)
                os.remove(filename)
            except:
                print(f"Could not extract {filename} to {path}. Please extract it manually.")
                print("The data might have been already extracted an is available in the test_data folder.")
            path = os.path.join(path, "test_data", "eval_data")
            return path

        return str(filename)

    def eval(self, dataset, *args, **kwargs):
        raise NotImplementedError

    def optimize(self, target_device):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    # Private methods ================================================================================
    def _fit(self,
             batch: List,
             return_losses: bool = False,
             replay_buffer: bool = False,
             learner: bool = False):
        """
        :param batch: list of tuples of (input, target)
        :type batch: List[Tuple[Dict, None]]
        :param return_losses: whether to return the losses
        :type return_losses: bool
        :param replay_buffer: whether to use replay buffer
        :type replay_buffer: bool
        :param learner: whether to use learner
        :type learner: bool
        """
        batch_size = len(batch)
        input_dict = self._input_formatter(batch, replay_buffer, learner=learner, height=self.height, width=self.width)
        # Adapt
        try:
            if return_losses:
                self.learner.adapt(input_dict, steps=5, return_loss=return_losses, batch_size=batch_size)
            else:
                self.learner.adapt(input_dict, steps=5, return_loss=return_losses, batch_size=batch_size)
            return True
        except:
            return False

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
        self.step += 1
        # Get the prediction
        prediction, losses = self.predictor.predict(input_dict, return_loss=return_losses)
        # Convert the prediction to opendr format
        depth, odometry = self._output_formatter(prediction)
        if not self.do_loop_closure:
            return (depth, odometry), losses
        else:
            if self.step == 1:
                self.pose_graph.add_vertex(0, np.eye(4), fixed=True)
            elif self.step > 1:
                odom_pose = self.pose_graph.get_pose(self.pose_graph.vertex_ids[-1]) @ odometry
                self.pose_graph.add_vertex(self.step, odom_pose)
                cov_matrix = np.eye(6)
                cov_matrix[2, 2] = .1
                cov_matrix[5, 5] = .1
                self.pose_graph.add_edge((self.pose_graph.vertex_ids[-2], self.step),
                                         odometry,
                                         information=np.linalg.inv(cov_matrix))
            optimized = False
            lc = False
            image = input_dict[(0, 'image')].squeeze()/255.0
            if self.step % self.key_frame_freq == 0:
                features = self.loop_closure_detection.add(self.step, image)
                self.online_dataset.add(self.step, image)
                if self.since_last_lc > self.lc_distance_poses:
                    lc_step_ids, distances = self.loop_closure_detection.search(self.step, features)
                    if len(lc_step_ids) > 0:
                        print(f'Loop closure detected at step {self.step}, candidates: {lc_step_ids}')
                        lc = True
                    for i, d in zip(lc_step_ids, distances):
                        lc_image = self.online_dataset.get(i)
                        lc_transformation, cov_matrix = self.predictor.predict_pose(image,
                                                                                    lc_image,
                                                                                    as_numpy=True)
                        cov_matrix = np.eye(6)
                        cov_matrix[2, 2] = .1
                        cov_matrix[5, 5] = .1
                        self.pose_graph.add_edge((self.step, int(i)),
                                                 lc_transformation,
                                                 information=.5 * np.linalg.inv(cov_matrix),
                                                 is_loop_closure=True)
                    if len(lc_step_ids) > 0:
                        self.pose_graph.optimize(max_iterations=100000, verbose=False)
                        optimized = True
            if optimized:
                self.since_last_lc = 0
            else:
                self.since_last_lc += 1
            return depth, np.expand_dims(odometry, axis=0), losses, lc, self.pose_graph

    @staticmethod
    def _input_formatter(batch: Tuple[Dict, None],
                         replay_buffer: bool = False,
                         learner: bool = False,
                         height: int = None,
                         width: int = None,)-> Union[List, Dict[Any, Tensor]]:
        """
        Format the input for the prediction
        :param batch: tuple of (input, target)
        :type batch: Tuple[Dict, None]
        :return: Either a list of input dicts or a single dictionary of input tensors
        :rtype: Union[List, Dict[Any, Tensor]]
        """
        if learner:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not (height and width):
                raise ValueError('Height and width must be provided for learner')

            input_dict = {(-1, 'image'): torch.zeros(size=(len(batch), 3, height, width), device=device),
                          (-1, 'distance'): torch.zeros(size=(len(batch), 1), device=device),
                          (0, 'image'): torch.zeros(size=(len(batch), 3, height, width), device=device),
                          (0, 'distance'): torch.zeros(size=(len(batch), 1), device=device),
                          (1, 'image'): torch.zeros(size=(len(batch), 3, height, width), device=device),
                          (1, 'distance'): torch.zeros(size=(len(batch), 1), device=device),
                          }
            for i, input in enumerate(batch):
                for frame_id in ([-1, 0, 1]):
                    image = input[(frame_id, 'image')]
                    distance = input[(frame_id, 'distance')]
                    if image.is_cuda and not torch.cuda.is_available():
                        image = image.cpu()
                        distance = distance.cpu()
                    if not image.is_cuda and torch.cuda.is_available():
                        image = image.cuda()
                        distance = distance.cuda()
                    input_dict[(frame_id, 'image')][i] = image
                    input_dict[(frame_id, 'distance')][i] = distance
            return input_dict

        if isinstance(batch, dict):
            inputs = batch
        else:
            inputs = batch[0]
        # Create a dictionary with frame ids as [-1, 0, 1]
        input_dict = {}
        if replay_buffer:
            return inputs
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
                    odometry = prediction[item][0, :].cpu().detach().numpy()
        return depth, odometry

    def _colorize_depth(self, depth):
        """
        Colorize the depth map
        """
        vmax = np.percentile(depth, 95)
        normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
        mapper = plt.cm.ScalarMappable(norm=normalizer, cmap="magma")
        colormapped_img = (mapper.to_rgba(depth.squeeze())[:, :, :3]*255).astype(np.uint8)
        colormapped_img = cv2.cvtColor(colormapped_img, cv2.COLOR_RGB2BGR)
        return Image(colormapped_img)
