# Copyright 2020-2022 OpenDR European Project
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
import torch.nn.functional as F
import continual as co
import pickle
import os
from opendr.engine import data
from opendr.engine.target import Category
from opendr.perception.activity_recognition.cox3d.algorithm.x3d import CoX3D
from opendr.perception.activity_recognition.utils.lightning import _LightningModuleWithCrossEntropy
from opendr.perception.activity_recognition.x3d.x3d_learner import X3DLearner
from pathlib import Path
from logging import getLogger
from typing import Union, List
import onnxruntime as ort


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
        temporal_window_size: int = None,
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
            weight_decay (float, optional): Weight decay used for optimization. Defaults to 1e-5.
            momentum (float, optional): Momentum used for optimization. Defaults to 0.9.
            drop_last (bool, optional): Drop last data point if a batch cannot be filled. Defaults to True.
            pin_memory (bool, optional): Pin memory in dataloader. Defaults to False.
            loss (str): Name of loss in torch.nn.functional to use. Defaults to "cross_entropy".
            num_workers (int, optional): Number of workers in dataloader. Defaults to 0.
            seed (int, optional): Random seed. Defaults to 123.
            num_classes (int, optional): Number of classes to predict among. Defaults to 400.
            temporal_window_size (int, optional): Size of the final global average pooling.
                If None, size will be automatically chosen according to the backbone. Defaults to None.
        """
        super().__init__(
            lr,
            iters,
            batch_size,
            optimizer,
            lr_schedule,
            backbone,
            network_head,
            checkpoint_after_iter,
            checkpoint_load_iter,
            temp_path,
            device,
            loss,
            weight_decay,
            momentum,
            drop_last,
            pin_memory,
            num_workers,
            seed,
            num_classes,
            *args,
            **kwargs,
        )
        self.temporal_window_size = temporal_window_size
        self._ort_state = None

    def init_model(self) -> CoX3D:
        """Initialise model with random parameters

        Returns:
            CoX3D: model
        """
        assert hasattr(self, "model_hparams"), "`self.model_hparams` not found. Did you forget to call `_load_hparams`?"
        self.model = CoX3D(
            dim_in=3,
            image_size=self.model_hparams["image_size"],
            temporal_window_size=getattr(self, "temporal_window_size", None) or self.model_hparams["frames_per_clip"],
            num_classes=self.num_classes,
            x3d_conv1_dim=self.model_hparams["conv1_dim"],
            x3d_conv5_dim=self.model_hparams["conv5_dim"],
            x3d_num_groups=self.model_hparams["num_groups"],
            x3d_width_per_group=self.model_hparams["width_per_group"],
            x3d_width_factor=self.model_hparams["width_factor"],
            x3d_depth_factor=self.model_hparams["depth_factor"],
            x3d_bottleneck_factor=self.model_hparams["bottleneck_factor"],
            x3d_use_channelwise_3x3x3=self.model_hparams["use_channelwise_3x3x3"],
            x3d_dropout_rate=self.model_hparams["dropout_rate"],
            x3d_head_activation=self.model_hparams["head_activation"],
            x3d_head_batchnorm=self.model_hparams["head_batchnorm"],
            x3d_fc_std_init=self.model_hparams["fc_std_init"],
            x3d_final_batchnorm_zero_init=self.model_hparams["final_batchnorm_zero_init"],
        ).to(device=self.device)
        self._plmodel = _LightningModuleWithCrossEntropy(self.model)
        return self.model

    def _map_state_dict(self, sd):
        if len(sd["head.lin_5.weight"]) > 3:
            sd["head.lin_5.weight"] = sd["head.lin_5.weight"].squeeze(-1).squeeze(-1)
        return sd

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

        if self._ort_session is not None and self._ort_state is not None:
            inputs = {
                "input": batch.cpu().detach().numpy(),
                **self._ort_state,
            }
            results, *next_state = self._ort_session.run(None, inputs)
            results = torch.tensor(results)
            self._ort_state = {k: v for k, v in zip(self._ort_state.keys(), next_state)}
        else:
            self.model.eval()
            results = self.model.forward_step(batch)
        if results is not None:
            results = [Category(prediction=int(r.argmax(dim=0)), confidence=F.softmax(r, dim=-1)) for r in results]
        return results

    def optimize(self, do_constant_folding=False):
        """Optimize model execution.
        This is accomplished by saving to the ONNX format and loading the optimized model.

        Args:
            do_constant_folding (bool, optional): Whether to optimize constants. Defaults to False.
        """

        if getattr(self.model, "_ort_session", None):
            logger.info("Model is already optimized. Skipping redundant optimization")
            return

        path = Path(self.temp_path or os.getcwd()) / "weights" / f"cox3d_{self.backbone}.onnx"
        if not path.exists():
            self._save_onnx(path, do_constant_folding)
        self._load_onnx(path)

    def _save_onnx(self, path: Union[str, Path], do_constant_folding=False, verbose=False):
        """Save model in the ONNX format

        Args:
            path (Union[str, Path]): Directory in which to save ONNX model
            do_constant_folding (bool, optional): Whether to optimize constants. Defaults to False.
        """
        path.parent.mkdir(exist_ok=True, parents=True)

        model = self.model.to(device="cpu")
        model.eval()

        # Prepare state
        state0 = None
        sample = self._example_input.repeat(self.batch_size, 1, 1, 1)
        with torch.no_grad():
            for _ in range(model.receptive_field):
                _, state0 = model._forward_step(sample, state0)
            _, state0 = model._forward_step(sample, state0)
            state0 = co.utils.flatten(state0)

            # Export to ONNX
            logger.info(f"Saving model to ONNX format at {str(path)}")
            co.onnx.export(
                model,
                (sample, *state0),
                path,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=do_constant_folding,
                verbose=verbose,
                opset_version=11,
            )

        # Save default state and name mappings for later use
        state_path = path.parent / f"cox3d_{self.backbone}_state.pickle"
        logger.info(f"Saving ONNX model states at {str(state_path)}")
        omodel = co.onnx.OnnxWrapper(self.model)
        state = {k: v.detach().numpy() for k, v in zip(omodel.state_input_names, state0)}
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

    def _load_onnx(self, path: Union[str, Path]):
        """Loads ONNX model into an onnxruntime inference session.

        Args:
            path (Union[str, Path]): Path to ONNX model
        """
        onnx_path = path
        state_path = path.parent / f"cox3d_{self.backbone}_state.pickle"

        logger.info(f"Loading ONNX runtime inference session from {str(onnx_path)}")
        self._ort_session = ort.InferenceSession(str(onnx_path))

        logger.info(f"Loading ONNX state from {str(state_path)}")
        with open(state_path, "rb") as f:
            self._ort_state = pickle.load(f)

    def _load_model_weights(self, weights_path: Union[str, Path]):
        """Load pretrained model weights

        Args:
            weights_path (Union[str, Path]): Path to model weights file.
                Type of file must be one of {".pyth", ".pth", ".onnx"}
        """
        weights_path = Path(weights_path)

        assert weights_path.is_file() and weights_path.suffix in {".pyth", ".pth", ".onnx"}, (
            f"weights_path ({str(weights_path)}) should be a .pth or .onnx file."
            "Pretrained weights can be downloaded using `self.download(...)`"
        )
        if weights_path.suffix == ".onnx":
            return self._load_onnx(weights_path)

        logger.debug(f"Loading model weights from {str(weights_path)}")

        # Check for configuration mismatches, loading only matching weights
        loaded_state_dict = torch.load(weights_path, map_location=torch.device(self.device))
        if "model_state" in loaded_state_dict:  # As found in the official pretrained X3D models
            loaded_state_dict = loaded_state_dict["model_state"]

        loaded_state_dict = self._map_state_dict(loaded_state_dict)

        self.model.load_state_dict(loaded_state_dict, strict=False, flatten=True)
        self.model.to(self.device)

        return self
