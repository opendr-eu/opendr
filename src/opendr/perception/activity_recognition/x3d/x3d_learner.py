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

from functools import partial
import json
import torch
import yaml
import os
from pathlib import Path
from opendr.engine.learners import Learner
from opendr.utils.io import bump_version
from torch import onnx
import onnxruntime as ort

from opendr.engine.data import Video
from opendr.engine.datasets import Dataset
from opendr.engine.target import Category

from opendr.perception.activity_recognition.x3d.algorithm.x3d import X3D
import pytorch_lightning as pl

from urllib.request import urlretrieve
from logging import getLogger
from typing import Any, Iterable, Union, Dict, List

logger = getLogger(__name__)

_MODEL_NAMES = {"xs", "s", "m", "l"}


class X3DLearner(Learner):
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
        *args,
        **kwargs,
    ):
        """Initialise the X3DLearner

        Args:
            lr (float, optional): Learning rate during optimization. Defaults to 1e-3.
            iters (int, optional): Number of epochs to train for. Defaults to 10.
            batch_size (int, optional): Dataloader batch size. Defaults to 64.
            optimizer (str, optional): Name of optimizer to use ("sgd" or "adam"). Defaults to "adam".
            lr_schedule (str, optional): Unused parameter. Defaults to "".
            network_head (str, optional): Head of network (only "classification" is currently available).
                Defaults to "classification".
            checkpoint_after_iter (int, optional): Unused parameter. Defaults to 0.
            checkpoint_load_iter (int, optional): Unused parameter. Defaults to 0.
            temp_path (str, optional): Path in which to store temporary files. Defaults to "".
            device (str, optional): Name of computational device ("cpu" or "cuda"). Defaults to "cuda".
            loss (str, optional): Loss function used during optimization. Defaults to "cross_entropy".
            weight_decay ([type], optional): Weight decay used for optimization. Defaults to 1e-5.
            momentum (float, optional): Momentum used for optimization. Defaults to 0.9.
            drop_last (bool, optional): Drop last data point if a batch cannot be filled. Defaults to True.
            pin_memory (bool, optional): Pin memory in dataloader. Defaults to False.
            num_workers (int, optional): Number of workers in dataloader. Defaults to 0.
            seed (int, optional): Random seed. Defaults to 123.
            num_classes (int, optional): Number of classes to predict among. Defaults to 400.
        """
        assert (
            backbone in _MODEL_NAMES
        ), f"Invalid model selected. Choose one of {_MODEL_NAMES}."
        assert network_head in {
            "classification"
        }, "Currently, only 'classification' head is supported."

        assert optimizer in {"sgd", "adam"}, "Supported optimizers are Adam and SGD."

        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(X3DLearner, self).__init__(
            lr=lr,
            iters=iters,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            backbone=backbone,
            network_head=network_head,
            temp_path=temp_path,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            device=device,
            threshold=0.0,
        )

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.seed = seed
        self.num_classes = num_classes
        self.loss = loss
        self.ort_session = None
        torch.manual_seed(self.seed)

        self._load_model_hparams(self.backbone)
        self.init_model()

    def _load_model_hparams(self, model_name: str = None) -> Dict[str, Any]:
        """Load hyperparameters for an X3D model

        Args:
            model_name (str, optional): Name of the model (one of {"xs", "s", "m", "l"}).
                If none, `self.backbon`e is used. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary with model hyperparameters
        """
        model_name = model_name or self.backbone
        assert (
            model_name in _MODEL_NAMES
        ), f"Invalid model selected. Choose one of {_MODEL_NAMES}."
        path = Path(__file__).parent / "hparams" / f"{model_name}.yaml"
        with open(path, "r") as f:
            self.model_hparams = yaml.load(f, Loader=yaml.FullLoader)
        return self.model_hparams

    def _load_model_weights(self, weights_path: Union[str, Path]):
        """Load pretrained model weights

        Args:
            weights_path (Union[str, Path]): Path to model weights file.
                Type of file must be one of {".pyth", ".pth", ".onnx"}
        """
        weights_path = Path(weights_path)

        assert (
            weights_path.is_file() and weights_path.suffix in {".pyth", ".pth", ".onnx"}
        ), (
            f"weights_path ({str(weights_path)}) should be a .pth or .onnx file."
            "Pretrained weights can be downloaded using `self.download(...)`"
        )
        if weights_path.suffix == ".onnx":
            return self._load_onnx(weights_path)

        logger.debug(f"Loading model weights from {str(weights_path)}")

        # Check for configuration mismatches, loading only matching weights
        new_model_state = self.model.state_dict()
        loaded_state_dict = torch.load(weights_path, map_location=torch.device(self.device))
        if "model_state" in loaded_state_dict:  # As found in the official pretrained X3D models
            loaded_state_dict = loaded_state_dict["model_state"]

        def size_ok(k):
            return new_model_state[k].size() == loaded_state_dict[k].size()

        to_load = {
            k: v for k, v in loaded_state_dict.items() if size_ok(k)
        }
        self.model.load_state_dict(to_load, strict=False)

        names_not_loaded = set(new_model_state.keys()) - set(to_load.keys())
        if len(names_not_loaded) > 0:
            logger.warning(f"Some model weight could not be loaded: {names_not_loaded}")

        return self

    def init_model(self) -> X3D:
        """Initialise model with random parameters

        Returns:
            X3D: model
        """
        assert hasattr(
            self, "model_hparams"
        ), "`self.model_hparams` not found. Did you forget to call `_load_hparams`?"
        self.model = X3D(
            dim_in=3,
            image_size=self.model_hparams["image_size"],
            frames_per_clip=self.model_hparams["frames_per_clip"],
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

    def save(self, path: Union[str, Path]):
        """Save model weights and metadata to path.

        Args:
            path (Union[str, Path]): Directory in which to save model weights and meta data.

        Returns:
            self
        """
        assert hasattr(
            self, "model"
        ), "Cannot save model because no model was found. Did you forget to call `__init__`?"

        root_path = Path(path)
        root_path.mkdir(parents=True, exist_ok=True)
        name = f"x3d_{self.backbone}"
        ext = ".onnx" if self.ort_session else ".pth"
        weights_path = bump_version(root_path / f"model_{name}{ext}")
        meta_path = bump_version(root_path / f"{name}.json")

        logger.info(f"Saving model weights to {str(weights_path)}")
        if self.ort_session:
            self._save_onnx(weights_path)
        else:
            torch.save(self.model.state_dict(), weights_path)

        logger.info(f"Saving meta-data to {str(meta_path)}")
        meta_data = {
            "model_paths": weights_path.name,
            "framework": "pytorch",
            "format": "pth",
            "has_data": False,
            "inference_params": {
                "backbone": self.backbone,
                "network_head": self.network_head,
                "threshold": self.threshold,
            },
            "optimized": bool(self.ort_session),
            "optimizer_info": {
                "lr": self.lr,
                "iters": self.iters,
                "batch_size": self.batch_size,
                "optimizer": self.optimizer,
                "checkpoint_after_iter": self.checkpoint_after_iter,
                "checkpoint_load_iter": self.checkpoint_load_iter,
                "loss": self.loss,
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
                "drop_last": self.drop_last,
                "pin_memory": self.pin_memory,
                "num_workers": self.num_workers,
                "seed": self.seed,
            },
        }
        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(meta_data, f, sort_keys=True, indent=4)

        return self

    def load(self, path: Union[str, Path]):
        """Load model.

        Args:
            path (Union[str, Path]): Path to metadata file in json format or path to model weights

        Returns:
            self
        """
        path = Path(path)

        # Allow direct loading of weights, omitting the metadatafile
        if path.suffix in {".pyth", ".pth", ".onnx"}:
            self._load_model_weights(path)
            return self
        if path.is_dir():
            path = path / f"x3d_{self.backbone}.json"
        assert (
            path.is_file() and path.suffix == ".json"
        ), "The provided metadata path should be a .json file"

        logger.debug(f"Loading X3DLearner metadata from {str(path)}")
        with open(path, "r") as f:
            meta_data = json.load(f)

        inference_params = meta_data["inference_params"]
        optimizer_info = meta_data["optimizer_info"]

        self.__init__(
            lr=optimizer_info["lr"],
            iters=optimizer_info["iters"],
            batch_size=optimizer_info["batch_size"],
            optimizer=optimizer_info["optimizer"],
            device=getattr(self, "device", "cpu"),
            threshold=inference_params["threshold"],
            backbone=inference_params["backbone"],
            network_head=inference_params["network_head"],
            loss=optimizer_info["loss"],
            checkpoint_after_iter=optimizer_info["checkpoint_after_iter"],
            checkpoint_load_iter=optimizer_info["checkpoint_load_iter"],
            weight_decay=optimizer_info["weight_decay"],
            momentum=optimizer_info["momentum"],
            drop_last=optimizer_info["drop_last"],
            pin_memory=optimizer_info["pin_memory"],
            num_workers=optimizer_info["num_workers"],
            seed=optimizer_info["seed"],
        )

        weights_path = path.parent / meta_data["model_paths"]
        self._load_model_weights(weights_path)

        return self

    @staticmethod
    def download(
        path: Union[str, Path], model_names: Iterable[str] = _MODEL_NAMES
    ):
        """Download pretrained X3D models

        Args:
            path (Union[str, Path], optional): Directory in which to store model weights. Defaults to None.
            model_names (Iterable[str], optional): iterable with model names to download.
                The iterable may contain {"xs", "s", "m", "l"}.
                Defaults to _MODEL_NAMES.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for m in model_names:
            assert m in _MODEL_NAMES
            filename = path / f"x3d_{m}.pyth"
            if filename.exists():
                logger.info(f"Skipping download of X3D-{m} (already exists at {str(filename)})")
            else:
                logger.info(f"Downloading pretrained X3D-{m} weight to {str(filename)}")
                urlretrieve(
                    url=f"https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_{m}.pyth",
                    filename=str(filename),
                )
                assert (
                    filename.is_file()
                ), f"Something wen't wrong when downloading {str(filename)}"

    def reset(self):
        pass

    def fit(self, dataset: Dataset, val_dataset: Dataset=None, epochs: int=None, steps: int=None, *args, **kwargs):
        """Fit the model to a dataset

        Args:
            dataset (Dataset): Training dataset
            val_dataset (Dataset, optional): Validation dataset.
                If none is given, validation steps are skipped. Defaults to None.
            epochs (int, optional): Number of epochs. If none is supplied, self.iters will be used. Defaults to None.
            steps (int, optional): Number of training steps to conduct. If none, this is determined by epochs. Defaults to None.
        """

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        ) if val_dataset else None

        optimisation_metric = "val/loss" if val_dataset else "train/loss"

        # Patch model optimizer
        assert self.optimizer in {"adam", "sgd"}, f"Invalid optimizer '{self.optimizer}'. Must be 'adam' or 'sgd'."
        if self.optimizer == "adam":
            Optimizer = partial(
                torch.optim.Adam,
                lr=self.lr,
                betas=(self.momentum, 0.999),
                weight_decay=self.weight_decay,
            )
        else:  # self.optimizer == "sgd":
            Optimizer = partial(
                torch.optim.Adam,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        def configure_optimizers():
            # nonlocal Optimizer, optimisation_metric
            optimizer = Optimizer(self.model.parameters())
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": optimisation_metric}

        self.model.configure_optimizers = configure_optimizers

        self.trainer = pl.Trainer(
            max_epochs=epochs or self.iters,
            gpus=1 if self.device == "cuda" else 0,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    save_top_k=1,
                    verbose=True,
                    monitor=optimisation_metric,
                    mode="min",
                    prefix="",
                )
            ],
            logger=_experiment_logger(),
        )
        self.trainer.limit_train_batches = steps or self.trainer.limit_train_batches
        self.trainer.limit_val_batches = steps or self.trainer.limit_val_batches

        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def eval(self, dataset: Dataset, steps: int=None) -> Dict[str, Any]:
        """Evaluate the model on the dataset

        Args:
            dataset (Dataset): Dataset on which to evaluate model
            steps (int, optional): Number of validation batches to evaluate.
                If None, all batches are evaluated. Defaults to None.

        Returns:
            Dict[str, Any]: Evaluation statistics
        """
        test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

        if not hasattr(self, "trainer"):
            self.trainer = pl.Trainer(
                gpus=1 if self.device == "cuda" else 0,
                logger=_experiment_logger(),
            )
        self.trainer.limit_test_batches = steps or self.trainer.limit_test_batches
        results = self.trainer.test(self.model, test_dataloader)
        results = {
            "accuracy": results[-1]["test/acc"],
            "loss": results[-1]["test/loss"],
        }
        return results

    def infer(self, batch: Union[Video, List[Video], torch.Tensor]) -> List[Category]:
        """Run inference on a batch of data

        Args:
            batch (torch.Tensor): Video or batch of video-clips.
                The video should have shape (3, T, H, W). If a batch is supplied, its shape should be (B, 3, T, H, W).

        Returns:
            torch.Tensor: Network output
        """
        # Cast to torch tensor
        if type(batch) is Video:
            batch = [batch]
        if type(batch) is list:
            batch = torch.stack([torch.tensor(v.data) for v in batch])

        batch = batch.to(device=self.device, dtype=torch.float)

        self.model.eval()
        results = self.model.forward(batch)
        results = [Category(prediction=int(r.argmax(dim=0)), confidence=r) for r in results]
        return results

    def optimize(self, do_constant_folding=False):
        """Optimize model execution.
        This is acoomplished by saving to the ONNX format and loading the optimized model.

        Args:
            do_constant_folding (bool, optional): Whether to optimize constants. Defaults to False.
        """

        if getattr(self.model, "rpn_ort_session", None):
            logger.info("Model is already optimized. Skipping redundant optimization")
            return

        path = Path(self.temp_path or os.getcwd()) / "weights" / f"x3d_{self.backbone}.onnx"
        if not path.exists():
            self._save_onnx(path, do_constant_folding)
        self._load_onnx(path)

    @property
    def _example_input(self):
        C = 3  # RGB
        T = self.model_hparams["frames_per_clip"]
        S = self.model_hparams["image_size"]
        return torch.randn(1, C, T, S, S).to(device=self.device)

    @_example_input.setter
    def _example_input(self):
        raise ValueError("_example_input is set thorugh 'frames_per_clip' 'image_size' in `self.model_hparams`")

    def _save_onnx(self, path: Union[str, Path], do_constant_folding=False, verbose=False):
        """Save model in the ONNX format

        Args:
            path (Union[str, Path]): Directory in which to save ONNX model
            do_constant_folding (bool, optional): Whether to optimize constants. Defaults to False.
        """
        path.parent.mkdir(exist_ok=True, parents=True)

        self.model.eval()
        self.model.to(device=self.device)

        logger.info(f"Saving model to ONNX format at {str(path)}")
        onnx.export(
            self.model,
            self._example_input,
            path,
            input_names=["video"],
            output_names=["classes"],
            do_constant_folding=do_constant_folding,
            verbose=verbose,
            opset_version=11,
        )

    def _load_onnx(self, path: Union[str, Path]):
        """Loads ONNX model into an onnxruntime inference session.

        Args:
            path (Union[str, Path]): Path to ONNX model
        """
        logger.info(f"Loading ONNX runtime inference session from {str(path)}")
        self.ort_session = ort.InferenceSession(str(path))


def _experiment_logger():
    return pl.loggers.TensorBoardLogger(save_dir=Path(os.getcwd()) / "logs", name="x3d")
