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
from engine.learners import Learner
from utils.io import bump_version

from engine.datasets import Dataset

# from engine.data import Video  # TODO: impl in engine.data
from perception.activity_recognition.x3d.modules.x3d import X3D
import pytorch_lightning as pl

# from engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve
from logging import getLogger
from typing import Any, Iterable, Union, Dict

logger = getLogger(__name__)

_MODEL_NAMES = {
    "xs",
    "s",
    "m",
    "l",
}


class X3DLearner(Learner):
    def __init__(
        self,
        lr=1e-3,
        iters=10,  # Epochs
        batch_size=64,
        optimizer="adam",
        lr_schedule="",  # Not used
        backbone="s",
        network_head="classification",  # Not used
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="",
        device="cuda",
        threshold=0.0,  # Not used
        loss="cross_entropy",
        rgb_mean=None,
        rgb_std=None,
        weight_decay=1e-5,
        momentum=0.9,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
        seed=123,
        num_classes=400,
        *args,
        **kwargs,
    ):
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
            threshold=threshold,
        )
        logger.debug("X3DLearner initialising")

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.seed = seed
        self.num_classes = num_classes
        self.loss = loss
        torch.manual_seed(self.seed)

        self.__load_model_hparams(self.backbone)
        self.init_model()
        logger.debug("X3DLearner initialised")

    def __load_model_hparams(self, model_name: str = None) -> Dict[str, Any]:
        model_name = model_name or self.backbone
        assert (
            model_name in _MODEL_NAMES
        ), f"Invalid model selected. Choose one of {_MODEL_NAMES}."
        path = Path(__file__).parent / "hparams" / f"{model_name}.yaml"
        with open(path, "r") as f:
            self.model_hparams = yaml.load(f, Loader=yaml.FullLoader)
        return self.model_hparams

    def load_model_weights(self, weights_path: str = None) -> Dict[str, Any]:
        weights_path = (
            Path(weights_path)
            if weights_path
            else Path(self.temp_path) / "weights" / f"x3d_{self.backbone}.pyth"
        )

        assert (
            weights_path.is_file() and weights_path.suffix in {".pyth", ".pth", ".onnx"}
        ), (
            f"weights_path ({str(weights_path)}) should be a .pth or .onnx file."
            "Pretrained weights can be downloaded using `X3DLearner.download(...)`"
        )
        logger.debug(f"Loading X3DLearner model weights from {str(weights_path)}")

        # Check for configuration mismatches, loading only matching weights
        new_model_state = self.model.state_dict()
        loaded_state_dict = torch.load(weights_path)

        def size_ok(k):
            return new_model_state[k].size() == loaded_state_dict[k].size()

        to_load = {
            k: v for k, v in loaded_state_dict.items() if size_ok(k)
        }
        self.model.load_state_dict(to_load, strict=False)

        names_not_loaded = set(new_model_state.keys()) - set(to_load.keys())
        if len(names_not_loaded) > 0:
            logger.warning(f"Some model weight could not be loaded: {names_not_loaded}")

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
        )
        return self.model

    def save(self, path: Union[str, Path]=None) -> "X3DLearner":
        """
        Save model weights and metadata to path.
        The saved model paths can be loaded using `self.load`.
        :param path: directory path for the model to be saved
        :type path: Union[str, Path]
        """

        assert hasattr(
            self, "model"
        ), "Cannot save model because no model was found. Did you forget to call `__init__`?"

        root_path = Path(path) if path else Path(self.temp_path)
        root_path.mkdir(parents=True, exist_ok=True)
        name = f"x3d_{self.backbone}"
        weights_path = bump_version(root_path / f"model_{name}.pth")
        meta_path = bump_version(root_path / f"{name}.json")

        logger.info(f"Saving X3DLearner model weights to {str(weights_path)}")
        torch.save(self.model.state_dict(), weights_path)

        logger.info(f"Saving X3DLearner meta-data to {str(meta_path)}")
        meta_data = {
            "model_paths": str(weights_path),
            "framework": "pytorch",
            "format": "pth",
            "has_data": False,
            "inference_params": {
                "backbone": self.backbone,
                "network_head": self.network_head,
                "threshold": self.threshold,
            },
            "optimized": False,
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

    def load(self, path: Union[str, Path]) -> "X3DLearner":
        """
        Loads the model from the provided, based on the metadata.json file included.
        :param path: path of the metadata json file or the folder containing it
        :type path: str
        """
        path = Path(path)
        if path.is_dir():
            path = path / f"x3d_{self.backbone}.json"
        assert (
            path.is_file() and path.suffix == ".json"
        ), "The provided path should be a .json file"

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
            # device=hparams["device"],
            threshold=inference_params["threshold"],
            backbone=inference_params["backbone"],
            network_head=inference_params["network_head"],
            # temp_path=hparams["temp_path"],
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

        weights_path = Path(meta_data["model_paths"])
        self.load_model_weights(weights_path)

        return self

    @staticmethod
    def download(
        path: Union[str, Path], model_weights: Iterable[str] = _MODEL_NAMES
    ):
        """Download pretrained X3D models

        Args:
            path (Union[str, Path], optional): Directory in which to store model weights. Defaults to None.
            model_weights (Iterable[str], optional): iterable with model names to download.
                The iterable may contain {"xs", "s", "m", "l"}.
                Defaults to _MODEL_NAMES.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for m in model_weights:
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
            pin_memory=self.num_workers > 1,
            drop_last=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.num_workers > 1,
            drop_last=True,
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
            limit_train_batches=steps or 1.0,
            limit_val_batches=steps or 1.0,
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
            logger=pl.loggers.TensorBoardLogger(save_dir=Path(os.getcwd()) / "logs", name="x3d"),
        )
        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def eval(self, dataset):
        """
        This method is used to evaluate the algorithm on a dataset
        and returns stats regarding the evaluation ran.

        :param dataset: Object that holds the dataset to evaluate the algorithm on
        :type dataset: Dataset class type
        :return: Returns stats regarding evaluation
        :rtype: dict
        """
        test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.num_workers > 1,
            drop_last=False,
        )

        if not hasattr(self, "trainer"):
            self.trainer = pl.Trainer(
                gpus=1 if self.device == "cuda" else 0,
                logger=pl.loggers.TensorBoardLogger(save_dir=Path(os.getcwd() / "logs"), name="x3d")
            )
        self.trainer.test(self.model, test_dataloader)

    def infer(self, point_clouds):
        ...

    def optimize(self, do_constant_folding=False):
        """
        Optimize method converts the model to ONNX format and saves the
        model in the parent directory defined by self.temp_path. The ONNX model is then loaded.
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        ...
