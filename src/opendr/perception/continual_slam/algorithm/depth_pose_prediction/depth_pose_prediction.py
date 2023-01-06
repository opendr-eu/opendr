from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F

from opendr.perception.continual_slam.algorithm.depth_pose_prediction.networks import ResnetEncoder, DepthDecoder, PoseDecoder
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.config import Config
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.dataset_config import DatasetConfig
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.utils import *



from opendr.perception.continual_slam.datasets import KittiDataset

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

        # if self.val_loader is None:
        #     self._create_dataset_loaders()
        self._set_eval()
        with torch.no_grad():
            outputs = self._make_prediction(batch)
        return outputs

    def _make_prediction(self, batch) -> Dict[Tensor, Any]:

        if len(batch) != 3:
            raise ValueError(f'Expected 3 images, but got {len(batch)}')

        for key, data in batch.items():
            if len(data.shape) == 3:
                data = data.unsqueeze(0)

            if type(data) != torch.Tensor:
                batch[key] = torch.from_numpy(data).to(self.device)
            else:
                data = data/255.0
                batch[key] = data.to(self.device)
        
        # Compute predictions
        outputs = {}
        outputs.update(self._predict_disparity(batch))
        outputs.update(self._predict_poses(batch))
        outputs.update(self._reconstruct_depth(outputs))

        return outputs

    def _reconstruct_depth(self,
                           outputs: Dict[Any, Tensor]) -> Dict[Any, Tensor]:
        """Reconstruct depth maps from disparity maps. """
        for scale in self.scales:
            disp = outputs[('disp', scale, 0)]
            disp = F.interpolate(disp, [self.height, self.width], mode='bilinear', align_corners=False)
            depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            outputs[('depth', scale, 0)] = depth
        return outputs

    def _predict_disparity(self, 
                           inputs: Dict[Any, Tensor],
                           frame_id: int = 0) -> Dict[Any, Tensor]:
        """Predict disparity for the current frame.
        @param inputs: dictionary containing the input images. The format is: {0: img_t1, -1: img_t2, 1: img_t0}
        @param frame_id: the frame id for which the disparity is predicted. Default is 0.
        @return: dictionary containing the disparity maps. The format is: {('disp', scale, frame_id): disp0, ...}        
        """

        features = self.models['depth_encoder'](inputs[frame_id])
        outputs = self.models['depth_decoder'](features)

        output = {}
        for scale in self.scales:
            output['disp', scale, frame_id] = outputs[("disp", scale)]
        return output

    def _predict_poses(self,
                       inputs: Dict[Any, Tensor]) -> Dict[Any, Tensor]:

        """Predict poses for the current frame and the previous one. 0 -> - 1 and 0 -> 1
        @param inputs: dictionary containing the input images. The format is: {0: img_t1, -1: img_t2, 1: img_t0}
        """

        outputs = {}
        pose_inputs_dict = inputs

        for frame_id in [-1, 1]:
            if frame_id == -1:
                pose_inputs = [pose_inputs_dict[frame_id], pose_inputs_dict[0]]
            else:
                pose_inputs = [pose_inputs_dict[0], pose_inputs_dict[frame_id]]
            pose_inputs = torch.cat(pose_inputs, 1)

            pose_features = [self.models['pose_encoder'](pose_inputs)]
            axis_angle, translation = self.models['pose_decoder'](pose_features)

            axis_angle, translation = axis_angle[:, 0], translation[:, 0]
            outputs[('axis_angle', 0, frame_id)] = axis_angle
            outputs[('translation', 0, frame_id)] = translation

            outputs[('cam_T_cam', 0, frame_id)] = transformation_from_parameters(axis_angle, 
                                                                              translation,
                                                                              invert=(frame_id == -1))
        return outputs

    def _create_dataset_loaders(self, training: bool = True, validation: bool = True) -> None:
        if self.dataset_type.lower() != 'kitti':
            raise ValueError(f'Unknown dataset type: {self.dataset_type}')
        if training:
            # self.train_loader = KittiDataset(self.dataset_path)
            raise NotImplementedError
        if validation:
            print('Loading validation dataset...')
            self.val_loader = KittiDataset(str(self.dataset_path))

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
                    self.optimizer.load_state_dict(optimizer_dict['optimizer']['param_groups'])
                    self.lr_scheduler.load_state_dict(optimizer_dict['scheduler'])
                    self.epoch = self.lr_scheduler.last_epoch
                    print(f'Restored optimizer and LR scheduler (resume from epoch {self.epoch}).')
                else:
                    self.optimizer.load_state_dict(optimizer_dict)
                    print('Restored optimizer (legacy mode).')
            except:  # pylint: disable=bare-except
                print('Cannot find matching optimizer weights, so the optimizer is randomly '
                      'initialized.')

    def _set_eval(self) -> None:
        """Set the model to evaluation mode
        """
        for model in self.models.values():
            if model is not None:
                model.eval()



# if __name__ == '__main__':
#     # Set local path
#     local_path = Path(__file__).parent.parent.parent / 'configs'
#     config = ConfigParser(local_path / 'singlegpu_kitti.yaml')

#     # Set up model
#     x = DepthPosePredictor(config.depth_pose, config.dataset)
#     x.load_model()
#     x._create_dataset_loaders(training=False, validation=True)
#     for batch in x.val_loader:
#         # This part should be removed afterwards, because the arriving input id's will be already 
#         # in the correct order and named as [-1, 0, 1], because ros node reciever will do that
#         # TODO: Remove this part
#         # ==================================================
#         inputs = prediction_input_formatter(batch)
#         # ==================================================
#         x.predict(inputs)

    