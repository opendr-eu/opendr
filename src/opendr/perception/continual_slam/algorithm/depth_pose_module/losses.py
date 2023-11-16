import torch
from torch import Tensor, nn
import torch.nn.functional as F

from typing import Dict, Any


def compute_reprojection_loss(ssim: nn.Module,
                              pred: Tensor,
                              target: Tensor,
                              ) -> Tensor:
    """
    Computes reprojection loss between a batch of predicted and target images
    This is the photometric error
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


def compute_smooth_loss(disp: Tensor,
                        img: Tensor,
                        mask: Tensor,
                        ) -> Tensor:
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


def compute_velocity_loss(inputs: Dict[Any, Tensor],
                          outputs: Dict[Any, Tensor],
                          distances: Dict[float, float],
                          device: torch.device,
                          ) -> Tensor:
    batch_size = inputs[0].shape[0]  # might be different from self.batch_size
    velocity_loss = torch.zeros(batch_size, device=device).squeeze()
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
        velocity_loss += F.l1_loss(pred_distance,
                                   gt_distance,
                                   reduction='none')  # separated by sample in batch
        num_frames += 1
    velocity_loss /= num_frames
    return velocity_loss
