import torch
from torch import nn


class Project3D(nn.Module):
    """
    Layer which projects 3D points into a camera with intrinsics K and at position T
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
