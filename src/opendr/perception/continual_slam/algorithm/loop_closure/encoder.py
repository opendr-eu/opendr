import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models, transforms


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LCFeatureEncoder:
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.num_features = 576

        self.model = models.mobilenet_v3_small(pretrained=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model.classifier = Identity()
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image: Tensor) -> Tensor:
        image = self.normalize(image)
        image = image.to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return features
