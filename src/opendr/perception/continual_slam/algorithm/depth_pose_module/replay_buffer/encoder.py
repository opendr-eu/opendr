import torch
from torch import Tensor
from torchvision import models, transforms
from feature_extractor import create_feature_extractor


class FeatureEncoder:
    def __init__(self, device: torch.device, num_features: int = 576):
        super().__init__()
        self.device = device

        self.model = models.mobilenet_v3_small(pretrained=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.model = create_feature_extractor(self.model, return_nodes=['flatten'])
        self.num_features = num_features

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image: Tensor) -> Tensor:
        image = self.normalize(image)
        image = image.to(self.device)
        with torch.no_grad():
            features = self.model(image)['flatten']
        return features