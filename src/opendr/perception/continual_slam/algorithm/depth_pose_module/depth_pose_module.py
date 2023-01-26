from typing import Any, Dict, Optional, Tuple

import math
import torch
import numpy as np
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from opendr.perception.continual_slam.algorithm.depth_pose_module.networks import (
    ResnetEncoder,
    DepthDecoder,
    PoseDecoder,
    )
from opendr.perception.continual_slam.algorithm.depth_pose_module.networks.layers import (
    SSIM,
    BackProjectDepth,
    Project3D,
    )
from opendr.perception.continual_slam.algorithm.depth_pose_module.losses import (
    compute_reprojection_loss,
    compute_smooth_loss,
    compute_velocity_loss,
)
from opendr.perception.continual_slam.algorithm.parsing.config import Config
from opendr.perception.continual_slam.algorithm.parsing.dataset_config import DatasetConfig
from opendr.perception.continual_slam.algorithm.depth_pose_module.utils import (
    transformation_from_parameters,
    disp_to_depth,
)
from opendr.perception.continual_slam.datasets import KittiDataset


class DepthPoseModule:
    """
    This class implements the DepthPosePredictor algorithm.
    """
    def __init__(self, config: Config, dataset_config: DatasetConfig, use_online: bool = False, mode: str = 'predictor'):

        # Parse the configuration parameters
        self._parse_configs(config, dataset_config)

        # Set the configuration parameters from input
        self.use_online = use_online
        self.mode = mode

        # Set static parameters
        self.num_pose_frames = 2
        self.camera_matrix = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.frame_ids = (0, -1, 1)

        # Set the device
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)

        # Create the model
        self._create_model()

        # Set the flag to indicate if the model is trained
        self.is_trained = False

    def adapt(self,
              inputs: Dict[Any, Tensor],
              online_index: int = 0,
              steps: int = 1,
              online_loss_weight: Optional[float] = None,
              use_expert: bool = True,
              do_adapt: bool = True,
              return_loss: bool = False
              ) -> Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]:
        """
        Adapt the model to a new task.
        :param inputs: A dictionary containing the inputs to the model.
        :type: Dict[Any, Tensor]
        :param online_index: The index of the online model in the batch. Defaults to 0.
        :type: int
        :param steps: The number of gradient steps to take. Defaults to 1.
        :type: int
        :param online_loss_weight: The weight to give to the online loss. Defaults to None.
        :type: float
        :param use_expert: Whether to use the expert model to compute the online loss. Defaults to True.
        :type: bool
        :param do_adapt: Whether to adapt the model. Defaults to True.
        :type: bool
        :param return_loss: Whether to return the loss. Defaults to False.
        :type: bool
        :return: A tuple containing the outputs and the loss.
        :rtype: Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]
        """
        if online_loss_weight is None:
            loss_weights = None
        elif self.batch_size == 1:
            loss_weights = torch.ones(1, device=self.device)
        else:
            loss_weights = torch.empty(self.batch_size, device=self.device)
            buffer_loss_weight = (1 - online_loss_weight) / (self.batch_size - 1)
            loss_weights[online_index] = online_loss_weight
            loss_weights[np.arange(self.batch_size) != online_index] = buffer_loss_weight

        if do_adapt:
            self._set_adapt(freeze_encoder=True)
        else:
            self._set_eval()
            steps = 1

        for _ in range(steps):
            self.optimizer.zero_grad()
            outputs, losses = self._process_batch(inputs, loss_weights)
            if do_adapt:
                losses['loss'].backward()
                self.optimizer.step()

        if self.batch_size != 1 and use_expert:
            # online_inputs = {key: value[online_index].unsqueeze(0) for key, value in inputs.items()}
            for _ in range(steps):
                self.online_optimizer.zero_grad()
                online_outputs, online_losses = self._process_batch(inputs, use_online=True)
                online_losses['loss'].backward()
                self.online_optimizer.step()
            outputs = online_outputs
            losses = online_losses
        if return_loss:
            return outputs, losses
        return outputs

    def predict(self,
                batch: Dict[Any, Tensor],
                return_loss: bool = False
                ) -> Tuple[Dict[Tensor, Any], Optional[Dict[Tensor, Any]]]:
        """
        Predict the output of a batch of inputs
        :param batch: A dictionary containing the inputs to the model.
        :type: Dict[Any, Tensor]
        :param return_loss: Whether to return the loss. Defaults to False.
        :type: bool
        :return: A dictionary containing the outputs of the model.
        :rtype: Dict[Tensor, Any]
        """
        self._set_eval()
        with torch.no_grad():
            outputs, losses = self._process_batch(batch)
        if return_loss:
            return outputs, losses
        return outputs

    def load_model(self, weights_folder: str = None, load_optimizer: bool = True) -> None:
        """
        Load the model from the checkpoint.
        :param load_optimizer: Whether to load the optimizer. Defaults to True.
        :type: bool
        """
        self._load_model(weights_folder, load_optimizer)

    def load_online_model(self, load_optimizer: bool = True) -> None:
        """
        Load the online model from the checkpoint.
        :param load_optimizer: Whether to load the optimizer. Defaults to True.
        :type: bool
        """
        self._load_online_model(load_optimizer)

    def create_dataset_loaders(self,
                               training: bool = False,
                               validation: bool = True,
                               ) -> None:
        """
        Create the dataset loaders.
        :param training: Whether to create the training dataset loader. Defaults to False.
        :type: bool
        :param validation: Whether to create the validation dataset loader. Defaults to True.
        :type: bool
        """
        self._create_dataset_loaders(training, validation)

    # Private methods --------------------------------------------------------------------------------------------
    def _parse_configs(self,
                       config: Config,
                       dataset_config: DatasetConfig
                       ) -> None:

        # Set the configuration parameters from dataset config
        self.dataset_type = dataset_config.dataset
        self.dataset_path = dataset_config.dataset_path
        self.height = dataset_config.height
        self.width = dataset_config.width

        # Set the configuration parameters from config
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
        self.batch_size = config.batch_size
        self.disparity_smoothness = config.disparity_smoothness
        self.velocity_loss_scaling = config.velocity_loss_scaling

    def _create_model(self) -> None:

        # Create a dictionary to store the models ==============================================================
        self.models = {}
        self.models['depth_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained)
        self.models['depth_decoder'] = DepthDecoder(self.models['depth_encoder'].num_ch_encoder,
                                                    self.scales)
        self.models['pose_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained,
                                                    self.num_pose_frames)
        self.models['pose_decoder'] = PoseDecoder(self.models['pose_encoder'].num_ch_encoder,
                                                  num_input_features=1,
                                                  num_frames_to_predict_for=2)

        self.online_models = {}
        if self.use_online:
            self.online_models['depth_encoder'] = ResnetEncoder(self.resnet,
                                                                self.resnet_pretrained)
            self.online_models['depth_decoder'] = DepthDecoder(self.models['depth_encoder'].num_ch_encoder,
                                                               self.scales)
            self.online_models['pose_encoder'] = ResnetEncoder(self.resnet,
                                                               self.resnet_pretrained,
                                                               self.num_pose_frames)
            self.online_models['pose_decoder'] = PoseDecoder(self.models['pose_encoder'].num_ch_encoder,
                                                             num_input_features=1,
                                                             num_frames_to_predict_for=2)
        self.backproject_depth = {}
        self.project_3d = {}
        self.backproject_depth_single = {}
        self.project_3d_single = {}
        for scale in self.scales:
            h = self.height // (2**scale)
            w = self.width // (2**scale)
            self.backproject_depth[scale] = BackProjectDepth(self.batch_size, h, w)
            self.project_3d[scale] = Project3D(self.batch_size, h, w)

            self.backproject_depth_single[scale] = BackProjectDepth(1, h, w)
            self.project_3d_single[scale] = Project3D(1, h, w)
        # ======================================================================================================

        # Structural similarity ===========================
        self.ssim = SSIM()
        self.ssim.to(self.device)
        # =================================================

        # Send everything to the GPU(s) ===================
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

        if self.use_online:
            self.online_parameters_to_train = []
            for m in self.online_models.values():
                m.to(self.device)
                self.online_parameters_to_train += list(m.parameters())
        for m in self.backproject_depth.values():
            m.to(self.device)
        for m in self.project_3d.values():
            m.to(self.device)
        for m in self.backproject_depth_single.values():
            m.to(self.device)
        for m in self.project_3d_single.values():
            m.to(self.device)
        # =================================================

        # Set up optimizer ================================
        self.optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_step_size, 0.1)
        self.epoch = 0
        if self.use_online:
            self.online_optimizer = optim.Adam(self.online_parameters_to_train, self.learning_rate)
        else:
            self.online_optimizer = None
        # =================================================

    def _process_batch(self,
                       batch: Dict[Any, Tensor],
                       loss_sample_weights: Optional[Tensor] = None,
                       use_online: bool = False
                       ) -> Dict[Tensor, Any]:

        if len(batch) != 6:
            raise ValueError(f'Expected 3 images, but got {len(batch)//2}')

        distances = {}
        inputs = {}
        for key, data in batch.items():
            if "distance" in key:
                distances[key[0]] = data.to(self.device)
                continue

            if len(data.shape) == 3:
                data = data.unsqueeze(0)

            if type(data) != torch.Tensor:
                inputs[key] = torch.from_numpy(data).to(self.device)
            else:
                data = data/255.0
                inputs[key[0]] = data.to(self.device)

        # Compute predictions
        # from PIL import Image as pil_image
        # for key, data in inputs.items():
        #     # x = inputs[key].cpu().numpy().squeeze(0)
        #     # print(x.shape, key)
        #     # img = pil_image.fromarray(x.transpose(1, 2, 0) * 255, 'RGB')
        #     img = ToPILImage()(inputs[key].cpu().squeeze(0))
        #     img.save(f'test_{key}.png')
        outputs = {}
        outputs.update(self._predict_disparity(inputs, use_online=use_online))
        outputs.update(self._predict_poses(inputs, use_online=use_online))
        outputs.update(self._reconstruct_images(inputs, outputs))

        if self.mode == 'learner':
            losses = self._compute_loss(inputs, outputs, distances, sample_weights=loss_sample_weights)
        elif self.mode == 'predictor':
            losses = None
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
        return outputs, losses

    def _compute_loss(self,
                      inputs: Dict[Any, Tensor],
                      outputs: Dict[Any, Tensor],
                      distances: Dict[Any, Tensor],
                      scales: Optional[Tuple[int, ...]]=None,
                      sample_weights: Optional[Tensor]=None
                      ) -> Dict[str, Tensor]:

        scales = self.scales if scales is None else scales
        if sample_weights is None:
            sample_weights = torch.ones(self.batch_size, device=self.device)/self.batch_size
        losses = {}
        total_loss = torch.zeros(1, device=self.device)
        scaled_inputs = self._create_scaled_inputs(inputs)

        for scale in self.scales:
            target = inputs[0]
            reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = outputs[('rgb', frame_id, scale)]
                reprojection_losses.append(compute_reprojection_loss(self.ssim, pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[frame_id]
                identity_reprojection_losses.append(compute_reprojection_loss(self.ssim, pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            # Add random numbers to break ties
            identity_reprojection_losses += torch.randn(identity_reprojection_losses.shape,
                                                        device=self.device) * 0.00001
            combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)
            to_optimize, _ = torch.min(combined, dim=1)
            reprojection_loss = (to_optimize.mean(2).mean(1) * sample_weights).sum()
            losses[f'reprojection_loss/scale_{scale}'] = reprojection_loss
            mask = torch.ones_like(outputs['disp', 0, scale], dtype=torch.bool, device=self.device)
            color = scaled_inputs['rgb', 0, scale]
            disp = outputs['disp', 0, scale]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = compute_smooth_loss(norm_disp, color, mask)
            smooth_loss = (smooth_loss * sample_weights).sum()
            losses[f'smooth_loss/scale_{scale}'] = smooth_loss
            # ==================================================

            regularization_loss = self.disparity_smoothness / (2**scale) * smooth_loss
            losses[f'reg_loss/scale_{scale}'] = regularization_loss
            # ==================================================

            loss = reprojection_loss + regularization_loss
            losses[f'depth_loss/scale_{scale}'] = loss
            total_loss += loss

        total_loss /= len(self.scales)
        losses['depth_loss'] = total_loss

        # Velocity supervision loss (scale independent) ====
        if self.velocity_loss_scaling is not None and self.velocity_loss_scaling > 0:
            velocity_loss = self.velocity_loss_scaling * \
                            compute_velocity_loss(inputs, outputs, distances, device=self.device)
            velocity_loss = (velocity_loss * sample_weights).sum()
            losses['velocity_loss'] = velocity_loss
            total_loss += velocity_loss
        # ==================================================

        losses['loss'] = total_loss

        if np.isnan(losses['loss'].item()):
            for k, v in losses.items():
                print(k, v.item())
            raise RuntimeError('NaN loss')

        return losses

    def _reconstruct_images(self,
                            inputs: Dict[Any, Tensor],
                            outputs: Dict[Any, Tensor],) -> Dict[Any, Tensor]:
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are added to the 'outputs' dictionary.
        """
        batch_size = outputs['disp', 0, self.scales[0]].shape[0]

        for scale in self.scales:
            # Upsample the disparity from scale to the target height x width
            disp = outputs[('disp', 0, scale)]
            disp = F.interpolate(disp, [self.height, self.width],
                                 mode='bilinear',
                                 align_corners=False)
            source_scale = 0

            depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            outputs[('depth', scale, 0)] = depth
            if self.mode == 'learner':
                camera_matrix, inv_camera_matrix = self._scale_camera_matrix(
                    self.camera_matrix, scale)
                camera_matrix = torch.Tensor(camera_matrix).to(self.device).unsqueeze(0)
                inv_camera_matrix = torch.Tensor(inv_camera_matrix).to(self.device).unsqueeze(0)

                for _, frame_id in enumerate(self.frame_ids[1:]):
                    T = outputs[('cam_T_cam', 0, frame_id)]

                    if batch_size == 1:
                        cam_points = self.backproject_depth_single[source_scale](
                            depth, inv_camera_matrix)
                        pixel_coordinates = self.project_3d_single[source_scale](
                            cam_points, camera_matrix, T)
                    else:
                        cam_points = self.backproject_depth[source_scale](depth,
                                                                          inv_camera_matrix)
                        pixel_coordinates = self.project_3d[source_scale](cam_points,
                                                                          camera_matrix,
                                                                          T)
                    # Save the warped image
                    outputs[('rgb', frame_id, scale)] = F.grid_sample(inputs[frame_id],
                                                                      pixel_coordinates,
                                                                      padding_mode='border',
                                                                      align_corners=True)
        return outputs

    def _predict_disparity(self,
                           inputs: Dict[Any, Tensor],
                           frame_id: int = 0,
                           use_online: bool = False) -> Dict[Any, Tensor]:
        """Predict disparity for the current frame.
        :param inputs: dictionary containing the input images. The format is: {0: img_t1, -1: img_t2, 1: img_t0}
        :param frame_id: the frame id for which the disparity is predicted. Default is 0.
        :return: dictionary containing the disparity maps. The format is: {('disp', frame_id, scale): disp0, ...}
        """
        if not use_online:
            features = self.models['depth_encoder'](inputs[frame_id])
            outputs = self.models['depth_decoder'](features)
        else:
            features = self.online_models['depth_encoder'](inputs[frame_id])
            outputs = self.online_models['depth_decoder'](features)

        output = {}
        for scale in self.scales:
            output['disp', frame_id, scale] = outputs[("disp", scale)]
        return output

    def _predict_poses(self,
                       inputs: Dict[Any, Tensor],
                       use_online: bool = False) -> Dict[Any, Tensor]:

        """Predict poses for the current frame and the previous one. 0 -> - 1 and 0 -> 1
        :param inputs: dictionary containing the input images. The format is: {0: img_t1, -1: img_t2, 1: img_t0}
        """

        outputs = {}
        pose_inputs_dict = inputs

        for frame_id in self.frame_ids[1:]:
            if frame_id == -1:
                pose_inputs = [pose_inputs_dict[frame_id], pose_inputs_dict[0]]
            else:
                pose_inputs = [pose_inputs_dict[0], pose_inputs_dict[frame_id]]
            pose_inputs = torch.cat(pose_inputs, 1)

            if use_online:
                pose_features = [self.online_models['pose_encoder'](pose_inputs)]
                axis_angle, translation = self.online_models['pose_decoder'](pose_features)
            else:
                pose_features = [self.models['pose_encoder'](pose_inputs)]
                axis_angle, translation = self.models['pose_decoder'](pose_features)

            axis_angle, translation = axis_angle[:, 0], translation[:, 0]
            outputs[('axis_angle', 0, frame_id)] = axis_angle
            outputs[('translation', 0, frame_id)] = translation

            outputs[('cam_T_cam', 0, frame_id)] = transformation_from_parameters(axis_angle,
                                                                                 translation,
                                                                                 invert=(frame_id == -1))
        return outputs

    def _scale_camera_matrix(self, camera_matrix: np.ndarray,
                             scale: int) -> Tuple[np.ndarray, np.ndarray]:
        scaled_camera_matrix = camera_matrix.copy()
        scaled_camera_matrix[0, :] *= self.width // (2**scale)
        scaled_camera_matrix[1, :] *= self.height // (2**scale)
        inv_scaled_camera_matrix = np.linalg.pinv(scaled_camera_matrix)
        return scaled_camera_matrix, inv_scaled_camera_matrix

    def _create_dataset_loaders(self, training: bool = True, validation: bool = True) -> None:
        if self.dataset_type.lower() != 'kitti':
            raise ValueError(f'Unknown dataset type: {self.dataset_type}')
        if training:
            # self.train_loader = KittiDataset(self.dataset_path)
            raise NotImplementedError
        if validation:
            print('Loading validation dataset...')
            self.val_loader = KittiDataset(str(self.dataset_path))

    def _load_model(self, weights_folder: str = None, load_optimizer: bool = True) -> None:
        """Load model(s) from disk
        """
        if weights_folder is not None:
            from pathlib import Path
            self.load_weights_folder = Path(weights_folder)
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
            print(path)
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


    def _load_online_model(self, load_optimizer: bool = True) -> None:
        """Load model(s) from disk
        """
        if self.load_weights_folder is None:
            print('Weights folder required to load the model is not specified.')
        if not self.load_weights_folder.exists():
            print(f'Cannot find folder: {self.load_weights_folder}')
        print(f'Load online model from: {self.load_weights_folder}')

        # Load the network weights
        for model_name, model in self.online_models.items():
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

        if load_optimizer:
            # Load the optimizer and LR scheduler
            optimizer_load_path = self.load_weights_folder / 'optimizer.pth'
            try:
                optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
                if 'optimizer' in optimizer_dict:
                    self.online_optimizer.load_state_dict(optimizer_dict['optimizer'])
                    print('Restored online optimizer')
                else:
                    self.online_optimizer.load_state_dict(optimizer_dict)
                    print('Restored online optimizer (legacy mode).')
            except:  # pylint: disable=bare-except
                print('Cannot find matching optimizer weights, so the optimizer is randomly '
                      'initialized.')

    # Setting the model to adapt, train or eval mode ==================================================
    def _set_adapt(self, freeze_encoder: bool = True) -> None:
        """
        Set all to train except for batch normalization (freeze parameters)
        Convert all models to adaptation mode: batch norm is in eval mode + frozen params
        Adapted from:
        https://github.com/Yevkuzn/CoMoDA/blob/main/code/CoMoDA.py
        """
        for model_name, model in self.models.items():
            model.eval()  # To set the batch norm to eval mode
            for name, param in model.named_parameters():
                if name.find('bn') != -1:
                    param.requires_grad = False  # Freeze batch norm
                if freeze_encoder and 'encoder' in model_name:
                    param.requires_grad = False

        for model_name, model in self.online_models.items():
            model.eval()  # To set the batch norm to eval mode
            for name, param in model.named_parameters():
                if name.find('bn') != -1:
                    param.requires_grad = False  # Freeze batch norm
                if freeze_encoder and 'encoder' in model_name:
                    param.requires_grad = False

    def _set_train(self) -> None:
        for m in self.models.values():
            if m is not None:
                m.train()

    def _set_eval(self) -> None:
        """Set the model to evaluation mode
        """
        for model in self.models.values():
            if model is not None:
                model.eval()

    # ==================================================================================================

    def _create_scaled_inputs(self, inputs):
        scaled_inputs = {}
        for scale in self.scales:
            exp_scale = 2 ** scale
            height = math.ceil(self.height / exp_scale)
            width = math.ceil(self.width / exp_scale)
            scaled_inputs[('rgb', 0, scale)] = F.interpolate(inputs[0],
                                                             [height, width],
                                                             mode='bilinear',
                                                             align_corners=True)
        return scaled_inputs
