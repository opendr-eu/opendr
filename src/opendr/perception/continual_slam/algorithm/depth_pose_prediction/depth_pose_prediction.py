from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F

from opendr.perception.continual_slam.algorithm.depth_pose_prediction.networks import (
    ResnetEncoder, 
    DepthDecoder, 
    PoseDecoder,
    BackprojectDepth,
    Project3D,
    SSIM,
    )
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.config import Config
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.dataset_config import DatasetConfig
from opendr.perception.continual_slam.algorithm.depth_pose_prediction.utils import *



from opendr.perception.continual_slam.datasets import KittiDataset

class DepthPosePredictor:
    def __init__(self, config: Config, dataset_config: DatasetConfig, use_online: bool = False) -> None:
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
        self.batch_size = config.batch_size
        self.disparity_smoothness = config.disparity_smoothness
        self.velocity_loss_scaling = config.velocity_loss_scaling


        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_type)
        self.num_pose_frames = 2
        self.is_trained = False

        self.use_online = use_online

        self.camera_matrix = np.array(
            [[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    
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

        self.online_models = {}
        if self.use_online:
            self.online_models['depth_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained)
            self.online_models['depth_decoder'] = DepthDecoder(self.models['depth_encoder'].num_ch_encoder,
                                                        self.scales)
            self.online_models['pose_encoder'] = ResnetEncoder(self.resnet, self.resnet_pretrained,
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
                self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
                self.project_3d[scale] = Project3D(self.batch_size, h, w)

                self.backproject_depth_single[scale] = BackprojectDepth(1, h, w)
                self.project_3d_single[scale] = Project3D(1, h, w)
            # =================================================

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
        if use_online:
            self.online_optimizer = optim.Adam(self.online_parameters_to_train, self.learning_rate)
        else:
            self.online_optimizer = None
        # =================================================

    def adapt(self,
              inputs: Dict[Any, Tensor],
              online_index: int = 0,
              steps: int = 1,
              online_loss_weight: Optional[float] = None,
              use_expert: bool = True,
              do_adapt: bool = True):

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
            outputs, losses = self._process_batch(inputs, loss_weights)
            if do_adapt:
                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()

        if self.batch_size != 1 and use_expert:
            # online_inputs = {key: value[online_index].unsqueeze(0) for key, value in inputs.items()}
            for _ in range(steps):
                online_outputs, online_losses = self._process_batch(inputs, use_online=True)
                self.online_optimizer.zero_grad()
                online_losses['loss'].backward()
                self.online_optimizer.step()
            outputs = online_outputs
            losses = online_losses

        return outputs, losses

        
    def predict(self, batch) -> Dict[Tensor, Any]:

        # if self.val_loader is None:
        #     self._create_dataset_loaders()
        self._set_eval()
        with torch.no_grad():
            outputs, losses = self._process_batch(batch)
        return outputs

    def _process_batch(self, 
                       batch: Dict[Any, Tensor],
                       loss_sample_weights: Optional[Tensor] = None,
                       use_online: bool = False) -> Dict[Tensor, Any]:

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
        outputs = {}
        outputs.update(self._predict_disparity(inputs, use_online=use_online))
        outputs.update(self._predict_poses(inputs, use_online=use_online))
        outputs.update(self._reconstruct_images(inputs, outputs))
        if self.use_online:
            losses = self._compute_loss(inputs, outputs, distances, sample_weights=loss_sample_weights)
        else:
            losses = None
        return outputs, losses

    def _compute_loss(self,
                      inputs: Dict[Any, Tensor],
                      outputs: Dict[Any, Tensor],
                      distances: Dict[Any, Tensor],
                      scales: Optional[Tuple[int, ...]] = None,
                      sample_weights: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Compute the reprojection and smoothness losses for a minibatch.
        """
        scales = self.scales if scales is None else scales
        if sample_weights is None:
            sample_weights = torch.ones(self.batch_size, device=self.device)/self.batch_size
        losses = {}
        total_loss = torch.zeros(1, device=self.device)
        scaled_inputs = self._create_scaled_inputs(inputs)
        for scale in self.scales:
            target = inputs[0]
            reprojection_losses = []
            for frame_id in [-1, 1]:
                pred = outputs[('rgb', frame_id, scale)]
                reprojection_losses.append(self._compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in [-1, 1]:
                pred = inputs[frame_id]
                identity_reprojection_losses.append(self._compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            # Add random numbers to break ties
            identity_reprojection_losses += torch.randn(identity_reprojection_losses.shape,
                                                        device=self.device) * 0.00001
            combined = torch.cat((identity_reprojection_losses, reprojection_losses), dim=1)
            to_optimize, _ = torch.min(combined, dim=1)
            reprojection_loss = (to_optimize.mean(2).mean(1) * sample_weights).sum()
            mask = torch.ones_like(outputs['disp', 0, scale], dtype=torch.bool, device=self.device)
            color = scaled_inputs['rgb', 0, scale]
            disp = outputs['disp', 0, scale]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = self._compute_smooth_loss(norm_disp, color, mask)
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
            velocity_loss = self.velocity_loss_scaling * self._compute_velocity_loss(
                inputs, outputs, distances)
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
            if self.use_online:
                camera_matrix = self.camera_matrix.copy()
                original_image_shape = inputs[0].shape
                camera_matrix[0, :] *= original_image_shape[-2]
                camera_matrix[1, :] *= original_image_shape[-1]
                inv_camera_matrix = np.linalg.pinv(camera_matrix)
                camera_matrix = torch.Tensor(camera_matrix).to(self.device)
                inv_camera_matrix = torch.Tensor(inv_camera_matrix).to(self.device)
                camera_matrix = camera_matrix.unsqueeze(0)
                inv_camera_matrix = inv_camera_matrix.unsqueeze(0)

                for i, frame_id in enumerate([-1, 1]):
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
                                                                        camera_matrix, T)
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

        for frame_id in [-1, 1]:
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

    def _create_scaled_inputs(self, inputs):
        scaled_inputs = {}
        scaled_hw = [(370, 1226), (185, 613), (93, 307), (47, 154)]
        for scale in self.scales:
            height, width = scaled_hw[scale]
            scaled_inputs[('rgb', 0, scale)] = F.interpolate(inputs[0], [height, width], mode='bilinear')
        return scaled_inputs

    def _compute_reprojection_loss(self,
                                   pred: Tensor,
                                   target: Tensor,
                                   ) -> Tensor:
        """
        Computes reprojection loss between a batch of predicted and target images
        This is the photometric error
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    @staticmethod
    def _compute_smooth_loss(disp: Tensor,
                             img: Tensor,
                             mask: Tensor) -> Tensor:
        """
        Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        grad_disp_x = torch.masked_select(grad_disp_x, mask[..., :-1])
        grad_disp_y = torch.masked_select(grad_disp_y, mask[..., :-1, :])

        batch_size = disp.shape[0]
        smooth_loss = torch.empty(batch_size, device=disp.device)
        for i in range(batch_size):
            _grad_disp_x = torch.masked_select(grad_disp_x[i, ...], mask[i, :, :, :-1])
            _grad_disp_y = torch.masked_select(grad_disp_y[i, ...], mask[i, :, :-1, :])
            smooth_loss[i] = _grad_disp_x.mean() + _grad_disp_y.mean()

        return smooth_loss


    def _compute_velocity_loss(
        self,
        inputs: Dict[Any, Tensor],
        outputs: Dict[Any, Tensor],
        distances: Dict[float, float]
    ) -> Tensor:
        batch_size = inputs[0].shape[0]  # might be different from self.batch_size
        velocity_loss = torch.zeros(batch_size, device=self.device).squeeze()
        num_frames = 0
        for frame in [0, -1, 1]:
            if frame == -1:
                continue
            if frame == 0:
                pred_translation = outputs[('translation', 0, -1)]
            else:
                pred_translation = outputs[('translation', 0, 1)]
            gt_distance = torch.abs(distances[frame]).squeeze()
            pred_distance = torch.linalg.norm(pred_translation, dim=-1).squeeze()
            velocity_loss += F.l1_loss(pred_distance, gt_distance,
                                       reduction='none')  # separated by sample in batch
            num_frames += 1
        velocity_loss /= num_frames
        return velocity_loss


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

    