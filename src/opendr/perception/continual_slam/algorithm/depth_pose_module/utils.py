from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from opendr.perception.continual_slam.algorithm.depth_pose_module.pytorch3d import (
    matrix_to_quaternion,
    quaternion_to_axis_angle,
)


def parameters_from_transformation(
    transformation: Tensor,
    as_numpy: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Convert a 4x4 transformation matrix to a translation vector and the axis angles
    """
    translation_vector = transformation[:, :, :3, 3]
    axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(transformation[:, :, :3, :3]))
    if as_numpy:
        translation_vector = translation_vector.squeeze().cpu().numpy()
        axis_angle = axis_angle.squeeze().cpu().numpy()
    return translation_vector, axis_angle


# -----------------------------------------------------------------------------

# The code below is adapted from:
# https://github.com/nianticlabs/monodepth2/blob/master/layers.py


def transformation_from_parameters(
    axis_angle: Tensor,
    translation: Tensor,
    invert: bool = False,
) -> Tensor:
    """Convert the network's (axis angle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axis_angle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector: Tensor) -> Tensor:
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(axis_angle: Tensor) -> Tensor:
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'axis_angle' has to be Bx1x3

    This is similar to the code below but has less rounding:
    from scipy.spatial.transform import Rotation
    Rotation.from_rotvec(axis_angle).as_matrix()
    """
    angle = torch.norm(axis_angle, 2, 2, True)
    axis = axis_angle / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((axis_angle.shape[0], 4, 4)).to(device=axis_angle.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def disp_to_depth(
    disp: Union[Tensor, np.ndarray],
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
) -> Union[Tensor, np.ndarray]:
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations' section of the
    monodepth2 paper.
    """
    # if sum(x is None for x in [min_depth, max_depth]) not in [0, 2]:
    #     raise ValueError('Either none or both of min_depth and max_depth must be None.')
    if min_depth is None and max_depth is None:
        depth = 1 / disp
    elif max_depth is None:
        depth = min_depth / disp
    elif min_depth is None:
        raise ValueError('min_depth is None')
    else:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
    return depth


# -----------------------------------------------------------------------------


def h_concat_images(im1: Image, im2: Image) -> Image:
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
