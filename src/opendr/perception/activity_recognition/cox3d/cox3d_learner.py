# Copyright 2021 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from opendr.engine import data
from opendr.engine.target import Category
from opendr.perception.activity_recognition.cox3d.algorithm.cox3d import CoX3D
from opendr.perception.activity_recognition.x3d.x3d_learner import X3DLearner

from logging import getLogger
from typing import Union, List

logger = getLogger(__name__)


class CoX3DLearner(X3DLearner):
    def __init__(
        self,
        lr=1e-3,
        iters=10,  # Epochs
        batch_size=64,
        optimizer="adam",
        lr_schedule="",
        backbone="s",
        network_head="classification",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="",
        device="cuda",
        loss="cross_entropy",
        weight_decay=1e-5,
        momentum=0.9,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
        seed=123,
        num_classes=400,
        temporal_window_size: int=None,
        *args,
        **kwargs,
    ):
        """Initialise the CoX3DLearner
        This learner wraps the Continual version of X3D, which makes predictions frame-by-frame rather than by clip.

        Args:
            lr (float, optional): Learning rate during optimization. Defaults to 1e-3.
            iters (int, optional): Number of epochs to train for. Defaults to 10.
            optimizer (str, optional): Name of optimizer to use ("sgd" or "adam"). Defaults to "adam".
            lr_schedule (str, optional): Unused parameter. Defaults to "".
            network_head (str, optional): Head of network (only "classification" is currently available).
                Defaults to "classification".
            checkpoint_after_iter (int, optional): Unused parameter. Defaults to 0.
            checkpoint_load_iter (int, optional): Unused parameter. Defaults to 0.
            temp_path (str, optional): Path in which to store temporary files. Defaults to "".
            device (str, optional): Name of computational device ("cpu" or "cuda"). Defaults to "cuda".
            weight_decay ([type], optional): Weight decay used for optimization. Defaults to 1e-5.
            momentum (float, optional): Momentum used for optimization. Defaults to 0.9.
            drop_last (bool, optional): Drop last data point if a batch cannot be filled. Defaults to True.
            pin_memory (bool, optional): Pin memory in dataloader. Defaults to False.
            num_workers (int, optional): Number of workers in dataloader. Defaults to 0.
            seed (int, optional): Random seed. Defaults to 123.
            num_classes (int, optional): Number of classes to predict among. Defaults to 400.
            temporal_window_size (int, optional): Size of the final global average pooling.
                If None, size will be automically chosen according to the backbone. Defaults to None.
        """
        super().__init__(
            lr, iters, batch_size, optimizer, lr_schedule, backbone, network_head, checkpoint_after_iter,
            checkpoint_load_iter, temp_path, device, loss, weight_decay, momentum, drop_last, pin_memory,
            num_workers, seed, num_classes, *args, **kwargs,
        )
        self.temporal_window_size = temporal_window_size

    def init_model(self) -> CoX3D:
        """Initialise model with random parameters

        Returns:
            CoX3D: model
        """
        assert hasattr(
            self, "model_hparams"
        ), "`self.model_hparams` not found. Did you forget to call `_load_hparams`?"
        self.model = CoX3D(
            dim_in=3,
            image_size=self.model_hparams["image_size"],
            frames_per_clip=getattr(self, "temporal_window_size", None) or self.model_hparams["frames_per_clip"],
            num_classes=self.num_classes,
            conv1_dim=self.model_hparams["conv1_dim"],
            conv5_dim=self.model_hparams["conv5_dim"],
            num_groups=self.model_hparams["num_groups"],
            width_per_group=self.model_hparams["width_per_group"],
            width_factor=self.model_hparams["width_factor"],
            depth_factor=self.model_hparams["depth_factor"],
            bottleneck_factor=self.model_hparams["bottleneck_factor"],
            use_channelwise_3x3x3=self.model_hparams["use_channelwise_3x3x3"],
            dropout_rate=self.model_hparams["dropout_rate"],
            head_activation=self.model_hparams["head_activation"],
            head_batchnorm=self.model_hparams["head_batchnorm"],
            fc_std_init=self.model_hparams["fc_std_init"],
            final_batchnorm_zero_init=self.model_hparams["final_batchnorm_zero_init"],
        ).to(device=self.device)
        return self.model

    @property
    def _example_input(self):
        C = 3  # RGB
        S = self.model_hparams["image_size"]
        return torch.randn(1, C, S, S).to(device=self.device)

    def infer(self, batch: Union[data.Image, List[data.Image], torch.Tensor]) -> List[Category]:
        """Run inference on a batch of data

        Args:
            batch (torch.Tensor): Image or batch of images.
                The image should have shape (3, H, W). If a batch is supplied, its shape should be (B, 3, H, W).

        Returns:
            List[target.Category]: List of output categories
        """
        # Cast to torch tensor
        if type(batch) is data.Image:
            batch = [batch]
        if type(batch) is list:
            batch = torch.stack([torch.tensor(v.data) for v in batch])

        batch = batch.to(device=self.device, dtype=torch.float)

        self.model.eval()
        results = self.model.forward(batch)
        results = [Category(prediction=int(r.argmax(dim=0)), confidence=r) for r in results]
        return results
