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

import continual as co
import json
import os
import pickle
import torch
import onnxruntime as ort
import pytorch_lightning as pl
import torch.nn.functional as F

from functools import partial
from pathlib import Path
from opendr.engine.target import Category
from opendr.engine.learners import Learner
from opendr.engine.helper.io import bump_version
from opendr.engine.datasets import Dataset
from opendr.engine.constants import OPENDR_SERVER_URL
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from urllib.request import urlretrieve

from logging import getLogger
from typing import Any, Union, Dict, List

from opendr.perception.skeleton_based_action_recognition.spatio_temporal_gcn_learner import (
    SpatioTemporalGCNLearner,
)
from opendr.perception.skeleton_based_action_recognition.algorithm.datasets.feeder import (
    Feeder,
)
from opendr.perception.skeleton_based_action_recognition.algorithm.models.co_stgcn import (
    CoStGcnMod,
)
from opendr.perception.skeleton_based_action_recognition.algorithm.models.co_agcn import (
    CoAGcnMod,
)
from opendr.perception.skeleton_based_action_recognition.algorithm.models.co_str import (
    CoSTrMod,
)


_MODEL_NAMES = {"costgcn", "costr", "coagcn"}

logger = getLogger(__name__)


class CoSTGCNLearner(Learner):
    def __init__(
        self,
        lr=1e-3,
        iters=10,  # Epochs
        batch_size=64,
        optimizer="adam",
        lr_schedule="",
        backbone="costgcn",
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
        num_classes=60,
        num_point=25,
        num_person=2,
        in_channels=3,
        graph_type="ntu",
        sequence_len: int = 300,
        *args,
        **kwargs,
    ):
        """Initialise the CoSTGCNLearnerLearner
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
            loss (str): Name of loss in torch.nn.functional to use. Defaults to "cross_entropy".
            weight_decay (float, optional): Weight decay used for optimization. Defaults to 1e-5.
            momentum (float, optional): Momentum used for optimization. Defaults to 0.9.
            drop_last (bool, optional): Drop last data point if a batch cannot be filled. Defaults to True.
            pin_memory (bool, optional): Pin memory in dataloader. Defaults to False.
            num_workers (int, optional): Number of workers in dataloader. Defaults to 0.
            seed (int, optional): Random seed. Defaults to 123.
            num_classes (int, optional): Number of classes to predict among. Defaults to 400.
            sequence_len (int, optional): Size of the final global average pooling. Defaults to 300.
        """
        super(CoSTGCNLearner, self).__init__(
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
        self._ort_session = None
        self._ort_state = None
        self.num_point = num_point
        self.num_person = num_person
        self.in_channels = in_channels
        self.graph_type = graph_type
        self.sequence_len = sequence_len

        if self.graph_type is None:
            raise ValueError(
                self.graph_type + "is not a valid graph type. Supported graphs: ntu, openpose"
            )
        if self.backbone is None or self.backbone not in _MODEL_NAMES:
            raise ValueError(
                self.backbone + f"is not a valid dataset name. Supported methods: {_MODEL_NAMES}"
            )

        pl.seed_everything(self.seed)
        self.init_model()

    def init_model(self) -> Union[CoStGcnMod, CoAGcnMod, CoSTrMod]:
        """Initialise model with random parameters

        Returns:
            Union[CoStGcnMod, CoAGcnMod, CoSTrMod]: model
        """
        Model = {
            "costgcn": CoStGcnMod,
            "coagcn": CoAGcnMod,
            "costr": CoSTrMod,
        }[self.backbone]

        self.model = Model(
            self.num_point,
            self.num_person,
            self.in_channels,
            self.graph_type,
            self.sequence_len,
            self.num_classes,
            self.loss,
        ).to(device=self.device)
        return self.model

    def download(
        self,
        dataset_name="nturgbd_cv",
        experiment_name="stgcn_nturgbd",
        path=None,
        method_name="costgcn",
        mode="pretrained",
        verbose=True,
        url=OPENDR_SERVER_URL + "perception/skeleton_based_action_recognition/",
        file_name="costgcn_ntu60_xview_joint.ckpt",
    ):
        if path is None:
            path = self.temp_path

        if mode == "pretrained":
            file_url = os.path.join(url, 'pretrained_models', method_name, file_name)
            target_path = Path(path) / method_name / dataset_name / file_name
            if target_path.exists():
                return str(target_path)
            target_path.parent.mkdir(exist_ok=True, parents=True)
            if verbose:
                print(f"Downloading {file_name} to {str(target_path)}")
            urlretrieve(file_url, str(target_path))
            return target_path

        # For dataset downloads, reuse the code from SpatioTemporalGCNLearner
        class _DownloadConfig:
            def __init__(self, parent_dir: str, dataset_name: str, experiment_name: str):
                self.parent_dir = parent_dir
                self.dataset_name = dataset_name
                self.experiment_name = experiment_name

        # Use download method from SpatioTemporalGCNLearner
        return SpatioTemporalGCNLearner.download(
            _DownloadConfig(self.temp_path, dataset_name, experiment_name),
            path,
            method_name,
            mode,
            verbose,
            url,
            file_name,
        )

    @staticmethod
    def _prepare_dataset(
        dataset,
        data_filename="train_joints.npy",
        labels_filename="train_labels.pkl",
        skeleton_data_type="joint",
        phase="train",
        verbose=True,
    ):
        if isinstance(dataset, ExternalDataset):
            if (
                dataset.dataset_type.lower() != "nturgbd" and
                dataset.dataset_type.lower() != "kinetics"
            ):
                raise UserWarning('dataset_type must be "NTURGBD or Kinetics"')
            # Get data and labels path
            data_path = os.path.join(dataset.path, data_filename)
            labels_path = os.path.join(dataset.path, labels_filename)
            if phase == "train":
                if dataset.dataset_type.lower() == "nturgbd":
                    random_choose = False
                    random_move = False
                    window_size = -1
                elif dataset.dataset_type.lower() == "kinetics":
                    random_choose = True
                    random_move = True
                    window_size = 150
            else:
                random_choose = False
                random_move = False
                window_size = -1

            if verbose:
                print("Dataset path is set. Loading feeder...")
            return Feeder(
                data_path=data_path,
                label_path=labels_path,
                random_choose=random_choose,
                random_move=random_move,
                window_size=window_size,
                skeleton_data_type=skeleton_data_type,
                data_name=dataset.dataset_type.lower(),
            )
        elif isinstance(dataset, DatasetIterator):
            return dataset

    def infer(self, batch: torch.Tensor) -> List[Category]:
        """Run inference on a batch of data

        Args:
            batch (torch.Tensor): batch of skeletons for a single time-step.
                The batch should have shape (C, V, S), (C, T, V, S), or (B, C, T, V, S).
                Here, B is the batch size, C is the number of input channels, V is the
                number of vertices, and S is the number of skeletons

        Returns:
            List[target.Category]: List of output categories
        """
        # Cast to torch tensor
        batch = batch.to(device=self.device, dtype=torch.float)
        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)  # (C, V, S) -> (B, C, V, S)
        if len(batch.shape) == 4:
            batch = batch.unsqueeze(2)  # (B, C, V, S) -> (B, C, T, V, S)

        if self._ort_session is not None and self._ort_state is not None:
            batch = batch.squeeze(2)  # (B, C, T, V, S) -> (B, C, V, S)
            inputs = {
                "input": batch.cpu().detach().numpy(),
                **self._ort_state,
            }
            results, *next_state = self._ort_session.run(None, inputs)
            results = torch.tensor(results)
            self._ort_state = {k: v for k, v in zip(self._ort_state.keys(), next_state)}
        else:
            self.model.eval()
            results = self.model.forward_steps(batch)
            if results is None:
                print("Warming model up prior to inference")
                _ = self.model.forward_steps(
                    batch.repeat(1, 1, self.model.receptive_field, 1, 1)
                )
                results = self.model.forward_steps(batch)

        results = [
            Category(prediction=int(r.argmax(dim=0)), confidence=F.softmax(r, dim=-1))
            for r in results
        ]
        return results

    def _load_model_weights(self, weights_path: Union[str, Path]):
        """Load pretrained model weights

        Args:
            weights_path (Union[str, Path]): Path to model weights file.
                Type of file must be one of {".pyth", ".pth", ".onnx"}
        """
        weights_path = Path(weights_path)

        assert weights_path.is_file() and weights_path.suffix in {
            ".pyth",
            ".pth",
            ".onnx",
            ".ckpt",
        }, (
            f"weights_path ({str(weights_path)}) should be a .pth or .onnx file."
            "Pretrained weights can be downloaded using `self.download(...)`"
        )
        if weights_path.suffix == ".onnx":
            return self._load_onnx(weights_path)

        logger.debug(f"Loading model weights from {str(weights_path)}")

        # Check for configuration mismatches, loading only matching weights
        new_model_state = self.model.state_dict()
        loaded_state_dict = torch.load(weights_path, map_location=torch.device(self.device))
        # As found in some pretrained models
        if "model_state" in loaded_state_dict:
            loaded_state_dict = loaded_state_dict["model_state"]
        # As found in PyTorch Lightning checkpoints
        if "state_dict" in loaded_state_dict:
            loaded_state_dict = loaded_state_dict["state_dict"]

        loaded_state_dict = self.model.map_state_dict(loaded_state_dict)

        def size_ok(k):
            return new_model_state[k].size() == loaded_state_dict[k].size()

        to_load = {k: v for k, v in loaded_state_dict.items() if size_ok(k)}
        self.model.load_state_dict(to_load, strict=False)

        names_not_loaded = set(new_model_state.keys()) - set(to_load.keys())
        if len(names_not_loaded) > 0:
            logger.warning(f"Some model weight could not be loaded: {names_not_loaded}")
        self.model.to(self.device)

        return self

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
        name = f"{self.backbone}"
        ext = ".onnx" if self._ort_session else ".pth"
        weights_path = bump_version(root_path / f"model_{name}{ext}")
        meta_path = bump_version(root_path / f"{name}.json")

        logger.info(f"Saving model weights to {str(weights_path)}")
        if self._ort_session:
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
                "num_classes": self.num_classes,
                "num_point": self.num_point,
                "num_person": self.num_person,
                "in_channels": self.in_channels,
                "graph_type": self.graph_type,
                "sequence_len": self.sequence_len,
            },
            "optimized": bool(self._ort_session),
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

        # Allow direct loading of weights, omitting the metadata file
        if path.suffix in {".pyth", ".pth", ".onnx", ".ckpt"}:
            self._load_model_weights(path)
            return self
        if path.is_dir():
            path = path / f"{self.backbone}.json"
        assert (
            path.is_file() and path.suffix == ".json"
        ), "The provided metadata path should be a .json file"

        logger.debug(f"Loading CoSTGCNLearner metadata from {str(path)}")
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
            num_classes=inference_params["num_classes"],
            num_point=inference_params["num_point"],
            num_person=inference_params["num_person"],
            in_channels=inference_params["in_channels"],
            graph_type=inference_params["graph_type"],
            sequence_len=inference_params["sequence_len"],
        )

        weights_path = path.parent / meta_data["model_paths"]
        self._load_model_weights(weights_path)

        return self

    def reset(self):
        pass

    def fit(
        self,
        dataset: Dataset,
        val_dataset: Dataset = None,
        epochs: int = None,
        steps: int = None,
        *args,
        **kwargs,
    ):
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
        val_dataloader = (
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )
            if val_dataset
            else None
        )

        optimisation_metric = "val/loss" if val_dataset else "train/loss"

        # Patch model optimizer
        assert self.optimizer in {
            "adam",
            "sgd",
        }, f"Invalid optimizer '{self.optimizer}'. Must be 'adam' or 'sgd'."
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
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": optimisation_metric,
            }

        self.model.configure_optimizers = configure_optimizers

        self.trainer = pl.Trainer(
            max_epochs=epochs or self.iters,
            gpus=1 if "cuda" in self.device else 0,
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
        self.model.to(self.device)

    def eval(self, dataset: Dataset, steps: int = None) -> Dict[str, Any]:
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
                gpus=1 if "cuda" in self.device else 0,
                logger=_experiment_logger(),
            )
        self.trainer.limit_test_batches = steps or self.trainer.limit_test_batches
        results = self.trainer.test(self.model, test_dataloader)
        results = {
            "accuracy": results[-1]["test/acc"],
            "loss": results[-1]["test/loss"],
        }
        return results

    def optimize(self, do_constant_folding=False):
        """Optimize model execution.
        This is accomplished by saving to the ONNX format and loading the optimized model.

        Args:
            do_constant_folding (bool, optional): Whether to optimize constants. Defaults to False.
        """

        if getattr(self.model, "ort_session", None):
            logger.info("Model is already optimized. Skipping redundant optimization")
            return

        path = Path(self.temp_path or os.getcwd()) / "weights" / f"{self.backbone}_weights.onnx"
        if not path.exists():
            self._save_onnx(path, do_constant_folding)
        self._load_onnx(path)

    @property
    def _example_input(self):
        return torch.randn(
            self.batch_size, self.in_channels, self.sequence_len, self.num_point, self.num_person
        ).to(device=self.device)

    @_example_input.setter
    def _example_input(self):
        raise ValueError(
            "_example_input is set through 'num_point', 'num_person', and 'in_channels' in constructor"
        )

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
        with torch.no_grad():
            for i in range(model.receptive_field):
                _, state0 = model._forward_step(self._example_input[:, :, i], state0)
            _, state0 = model._forward_step(self._example_input[:, :, -1], state0)
            state0 = co.utils.flatten(state0)

            # Export to ONNX
            logger.info(f"Saving model to ONNX format at {str(path)}")
            co.onnx.export(
                model,
                (self._example_input[:, :, -1], *state0),
                path,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=do_constant_folding,
                verbose=verbose,
                opset_version=12,
            )

        # Save default state and name mappings for later use
        state_path = path.parent / f"{self.backbone}_state.pickle"
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
        state_path = path.parent / f"{self.backbone}_state.pickle"

        logger.info(f"Loading ONNX runtime inference session from {str(onnx_path)}")
        self._ort_session = ort.InferenceSession(str(onnx_path))

        logger.info(f"Loading ONNX state from {str(state_path)}")
        with open(state_path, "rb") as f:
            self._ort_state = pickle.load(f)


def _experiment_logger():
    return pl.loggers.TensorBoardLogger(save_dir=Path(os.getcwd()) / "logs", name="costgcn")
