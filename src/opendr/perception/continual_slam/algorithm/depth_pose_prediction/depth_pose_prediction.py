import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn, optim, Tensor

from opendr.perception.continual_slam.algorithm.depth_pose_prediction.networks import ResnetEncoder, DepthDecoder, PoseDecoder
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.config import Config
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.dataset_config import DatasetConfig
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.utils import *

from opendr.perception.continual_slam.configs.config_parser import ConfigParser


class DepthPosePredictor:
    def __init__(self, config: Config, dataset_config: DatasetConfig) -> None:
        self.dataset_type = dataset_config.dataset
        self.dataset_path = dataset_config.dataset_path
        self.height = dataset_config.height
        self.width = dataset_config.width

        self.resnet = config.resnet
        self.resnet_pretrained = config.resnet_pretrained
        self.scales = config.scales
        self.gpu_ids = config.gpu_ids
        self.multiple_gpus = config.multiple_gpus
        self.learning_rate = config.learning_rate
        self.scheduler_step_size = config.scheduler_step_size
        self.load_weights_folder = config.load_weights_folder
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth

        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.num_pose_frames = 2
        self.is_trained = False

    
        # Create a dictionary to store the models
        self.models = {}
        self.models['depth_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained)
        self.models['depth_decoder'] = DepthDecoder(self.models['depth_encoder'].num_ch_encoder,
                                                    self.scales)
        self.models['pose_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained,
                                                    self.num_pose_frames)
        self.models['pose_decoder'] = PoseDecoder(self.models['pose_encoder'].num_ch_encoder,
                                                  num_input_features=1,
                                                  num_frames_to_predict_for=2)
        
        if 'cuda' in self.device.type:
            print(f'Selected GPUs: {list(self.gpu_ids)}')
        if self.multiple_gpus:
            for model_name, m in self.models.items():
                if m is not None:
                    self.models[model_name] = nn.DataParallel(m, device_ids=self.gpu_ids)

        self.parameters_to_train = []
        for model_name, m in self.models.items():
            if m is not None:
                m.to(self.device)
                self.parameters_to_train += list(m.parameters())

        # Set up optimizer ================================
        self.optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_step_size, 0.1)
        self.epoch = 0
        self.online_optimizer = None
        # =================================================

        # Construct dataset loaders =======================
        self.train_loader, self.val_loader = None, None
        # =================================================

    def predict(self, batch) -> Dict[Tensor, Any]:

        self._set.eval()
        with torch.no_grad():
            outputs = self._make_prediction(batch)
        return outputs

    def _make_prediction(self, batch) -> Dict[Tensor, Any]:

        pass

    def _make_predict_one_image(self, image):
        with torch.no_grad():
            image = image.to(self.device)
            features = self.models['depth_encoder'](image)
            disp  = self.models['depth_decoder'](features)[('disp', 0)]

            depth = disp_to_depth(disp, self.min_depth, self.max_depth)
        return depth

    def _make_predict_two_images(self, image0, image1):

        if len(image0.shape) == 3:
            image0 = image0.unsqueeze(dim=0)
        if len(image1.shape) == 3:
            image1 = image1.unsqueeze(dim=0)

        with torch.no_grad():
            image0 = image0.to(self.device)
            image1 = image1.to(self.device)
            features_0 = self.models['depth_encoder'](image0)
            disp_0 = self.models['depth_decoder'](features_0)[('disp', 0)]
            features_1 = self.models['depth_encoder'](image1)
            disp_1 = self.models['depth_decoder'](features_1)[('disp', 0)]
            depth_0 = disp_to_depth(disp_0, self.min_depth, self.max_depth)
            depth_1 = disp_to_depth(disp_1, self.min_depth, self.max_depth)

            pose_inputs = torch.cat((image0, image1), 1)
            pose_features = self.models['pose_encoder'](pose_inputs)
            
            pose_inputs = torch.cat(image1)



    def load_model(self, load_optimizer: bool = True) -> None:
        """Load model(s) from disk
        """
        if self.load_weights_folder is None:
            print('Weights folder required to load the model is not specified.')
        if not self.load_weights_folder.exists():
            print(f'Cannot find folder: {self.load_weights_folder}')
        print(f'Load model from: {self.load_weights_folder}')

        # Load the network weights
        for model_name, model in self.models.items():
            if model is None:
                continue
            path = self.load_weights_folder / f'{model_name}.pth'
            pretrained_dict = torch.load(path, map_location=self.device)
            if isinstance(model, nn.DataParallel):
                model_dict = model.module.state_dict()
            else:
                model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict.keys()) == 0:
                raise RuntimeError(f'No fitting weights found in: {path}')
            model_dict.update(pretrained_dict)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(model_dict)
            else:
                model.load_state_dict(model_dict)
        self.is_trained = True

        if load_optimizer:
            # Load the optimizer and LR scheduler
            optimizer_load_path = self.load_weights_folder / 'optimizer.pth'
            try:
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                if 'optimizer' in optimizer_dict:
                    self.optimizer.load_state_dict(optimizer_dict['optimizer'])
                    self.lr_scheduler.load_state_dict(optimizer_dict['scheduler'])
                    self.epoch = self.lr_scheduler.last_epoch
                    print(f'Restored optimizer and LR scheduler (resume from epoch {self.epoch}).')
                else:
                    self.optimizer.load_state_dict(optimizer_dict)
                    print('Restored optimizer (legacy mode).')
            except:  # pylint: disable=bare-except
                print('Cannot find matching optimizer weights, so the optimizer is randomly '
                      'initialized.')

if __name__ == '__main__':
    # Set local path
    local_path = Path(__file__).parent.parent.parent / 'configs'
    config = ConfigParser(local_path / 'singlegpu_kitti.yaml')

    # Set up model
    x = DepthPosePredictor(config.depth_pose, config.dataset)
    x.load_model()
    