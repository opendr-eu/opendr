# Adapted from:
# https://github.com/nianticlabs/monodepth2/blob/master/layers.py#L64

import numpy as np
import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_reflection: bool = True,
    ) -> None:
        super().__init__()

        if use_reflection:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size: int, height: int, width: int) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size: int, height: int, width: int, eps: float = 1e-7) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pixel_coordinates = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pixel_coordinates = pixel_coordinates.view(self.batch_size, 2, self.height, self.width)
        pixel_coordinates = pixel_coordinates.permute(0, 2, 3, 1)
        pixel_coordinates[..., 0] /= self.width - 1
        pixel_coordinates[..., 1] /= self.height - 1
        pixel_coordinates = (pixel_coordinates - 0.5) * 2
        return pixel_coordinates


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self) -> None:
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
