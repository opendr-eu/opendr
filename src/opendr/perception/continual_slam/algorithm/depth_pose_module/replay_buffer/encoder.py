import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models, transforms


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatureEncoder:
    def __init__(self, device: torch.device, num_features: int = 576):
        super().__init__()
        self.device = device

        self.model = models.mobilenet_v3_small(pretrained=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.num_features = num_features
        self.model.classifier = Identity()
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image: Tensor) -> Tensor:
        image = self.normalize(image)
        image = image.to(self.device).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image)
        return features
