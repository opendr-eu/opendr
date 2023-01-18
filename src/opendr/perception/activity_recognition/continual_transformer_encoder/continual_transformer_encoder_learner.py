# Copyright 2020-2023 OpenDR European Project
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
import os
import pickle
from pathlib import Path
from opendr.engine.learners import Learner
from opendr.engine.helper.io import bump_version
import onnxruntime as ort
from collections import OrderedDict

from opendr.engine.data import Timeseries, Vector
from opendr.engine.datasets import Dataset
from opendr.engine.target import Category

from logging import getLogger
from typing import Any, Union, Dict

import pytorch_lightning as pl
import continual as co
from continual import onnx

logger = getLogger(__name__)


class CoTransEncLearner(Learner):
    def __init__(
        self,
        lr=1e-2,
        iters=10,  # Epochs
        batch_size=64,
        optimizer="sgd",
        lr_schedule="",
        network_head="classification",
        num_layers=1,  # 1 or 2
        input_dims=1024,
        hidden_dims=1024,
        sequence_len=64,
        num_heads=8,
        dropout=0.1,
        num_classes=22,
        positional_encoding_learned=False,
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="",
        device="cuda",
        loss="cross_entropy",
        weight_decay=1e-4,
        momentum=0.9,
        drop_last=True,
        pin_memory=False,
        num_workers=0,
        seed=123,
        *args,
        **kwargs,
    ):
        """Initialise the CoTransLearner
        This learner wraps the Continual Transformer Encoder with Recycling Positional Encoding
        which optimises token-by-token predictions for temporal sequences.
        It was proposed for Online Action Detection in
        "L. Hedegaard, A. Bakhtiarnia, and A. Iosifidis: 'Continual Transformers: Redundancy-Free
        Attention for Online Inference', 2022"
        https://arxiv.org/abs/2201.06268

        Args:
            lr (float, optional): Learning rate during optimization. Defaults to 1e-3.
            iters (int, optional): Number of epochs to train for. Defaults to 10.
            optimizer (str, optional): Name of optimizer to use ("sgd" or "adam"). Defaults to "adam".
            lr_schedule (str, optional): Schedule for training the model. Only "ReduceLROnPlateau" is available currently.
            network_head (str, optional): Head of network (only "classification" is currently available).
                Defaults to "classification".
            num_layers (int, optional): Number of Transformer Encoder layers (1 or 2). Defaults to 1.
            input_dims (int, optional): Input token dimension. Defaults to 1024.
            hidden_dims (int, optional): Hidden projection dimension. Defaults to 1024.
            sequence_len (int, optional): Length of token sequence to consider. Defaults to 64.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            num_classes (int, optional): Number of classes to predict. Defaults to 22.
            positional_encoding_learned (bool, optional): Whether positional encoding is learned. Defaults to False.
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
        """
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        assert optimizer in {"sgd", "adam"}, "Supported optimizers are Adam and SGD."
        assert network_head in {
            "classification"
        }, "Currently, only 'classification' head is supported."

        super(CoTransEncLearner, self).__init__(
            lr=lr,
            iters=iters,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            network_head=network_head,
            temp_path=temp_path,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            device=device,
        )

        assert num_layers in {
            1,
            2,
        }, "Only 1 or 2 Transformer Encoder layers are supported."
        self._num_layers = num_layers
        self._positional_encoding_learned = positional_encoding_learned
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._sequence_len = sequence_len
        self._num_heads = num_heads
        self._dropout = dropout
        self._num_classes = num_classes
        self._weight_decay = weight_decay
        self._momentum = momentum
        self._drop_last = drop_last
        self._pin_memory = pin_memory
        self._num_workers = num_workers
        self._loss = loss
        self._ort_session = None
        self._seed = seed
        torch.manual_seed(self._seed)

        self.init_model()

    def init_model(self) -> torch.nn.Module:
        """Initialise model with random parameters

        Returns:
            torch.nn.Module: Continual Transformer Encoder with
                Recycling Positional Encoding
        """
        pos_enc = co.RecyclingPositionalEncoding(
            embed_dim=self._input_dims,
            num_embeds=self._sequence_len * 2 - 1,
            learned=self._positional_encoding_learned,
        )
        trans_enc = co.TransformerEncoder(
            co.TransformerEncoderLayerFactory(
                d_model=self._input_dims,
                nhead=self._num_heads,
                dim_feedforward=self._hidden_dims,
                dropout=self._dropout,
                sequence_len=self._sequence_len,
            ),
            num_layers=self._num_layers,
        )
        lin = co.Linear(self._input_dims, self._num_classes, channel_dim=-1)

        self.model = co.Sequential(
            OrderedDict(
                [
                    ("pos_enc", pos_enc),
                    ("trans_enc", trans_enc),
                    (
                        "select",
                        co.Lambda(
                            fn=lambda x: x[:, :, -1],
                            forward_step_only_fn=lambda x: x,
                            takes_time=True,
                        ),
                    ),
                    ("lin", lin),
                ]
            )
        )

        class AddIfTraining:
            def __init__(slf, val: int):
                slf.val = val

            def __add__(slf, other: int):
                return other + (slf.val if self.model.training else 0)

            def __radd__(slf, other: int):
                return slf.__add__(other)

        self.model[0].forward_update_index_steps = AddIfTraining(1)
        self.model = self.model.to(device=self.device)

        self._plmodel = _LightningModuleWithCrossEntropy(self.model)
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
        name = "cotransenc_weights"
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
                "network_head": self._network_head,
                "num_layers": self._num_layers,
                "input_dims": self._input_dims,
                "hidden_dims": self._hidden_dims,
                "sequence_len": self._sequence_len,
                "num_heads": self._num_heads,
                "dropout": self._dropout,
                "num_classes": self._num_classes,
                "positional_encoding_learned": self._positional_encoding_learned,
            },
            "optimized": bool(self._ort_session),
            "optimizer_info": {
                "lr": self.lr,
                "iters": self.iters,
                "batch_size": self.batch_size,
                "optimizer": self.optimizer,
                "checkpoint_after_iter": self.checkpoint_after_iter,
                "checkpoint_load_iter": self.checkpoint_load_iter,
                "loss": self._loss,
                "weight_decay": self._weight_decay,
                "momentum": self._momentum,
                "drop_last": self._drop_last,
                "pin_memory": self._pin_memory,
                "num_workers": self._num_workers,
                "seed": self._seed,
                "dropout": self._dropout,
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
            path = path / "cotransenc_weights.json"
        assert (
            path.is_file() and path.suffix == ".json"
        ), "The provided metadata path should be a .json file"

        logger.debug(f"Loading CoTransEnc metadata from {str(path)}")
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
            network_head=inference_params["network_head"],
            num_layers=inference_params["num_layers"],
            input_dims=inference_params["input_dims"],
            hidden_dims=inference_params["hidden_dims"],
            sequence_len=inference_params["sequence_len"],
            num_heads=inference_params["num_heads"],
            num_classes=inference_params["num_classes"],
            positional_encoding_learned=inference_params["positional_encoding_learned"],
            dropout=optimizer_info["dropout"],
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
            return self._load_onnx(weights_path.parent)

        logger.debug(f"Loading model weights from {str(weights_path)}")

        loaded_state_dict = torch.load(weights_path, map_location=torch.device(self.device))
        self.model.load_state_dict(loaded_state_dict, strict=False)

        return self

    @staticmethod
    def download(path: Union[str, Path]):
        """Download pretrained models. As this module

        Args:
            path (Union[str, Path], optional): Directory in which to store model weights. Defaults to None.
            model_names (Iterable[str], optional): iterable with model names to download.
                The iterable may contain {"xs", "s", "m", "l"}.
                Defaults to _MODEL_NAMES.
        """
        raise NotImplementedError(
            "No pretrained models available. Please train your own model using the `fit` function."
        )

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
            num_workers=self._num_workers,
            shuffle=True,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )
        val_dataloader = (
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                pin_memory=self._pin_memory,
                drop_last=self._drop_last,
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
                betas=(self._momentum, 0.999),
                weight_decay=self._weight_decay,
            )
        else:  # self.optimizer == "sgd":
            Optimizer = partial(
                torch.optim.SGD,
                lr=self.lr,
                momentum=self._momentum,
                weight_decay=self._weight_decay,
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

        self._plmodel.configure_optimizers = configure_optimizers

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

        self.trainer.fit(self._plmodel, train_dataloader, val_dataloader)
        # self.model.to(self.device)

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
            num_workers=self._num_workers,
            shuffle=False,
            pin_memory=self._pin_memory,
            drop_last=False,
        )

        if not hasattr(self, "trainer"):
            self.trainer = pl.Trainer(
                gpus=1 if "cuda" in self.device else 0,
                logger=_experiment_logger(),
            )
        self.trainer.limit_test_batches = steps or self.trainer.limit_test_batches
        results = self.trainer.test(self._plmodel, test_dataloader)
        results = {
            "accuracy": results[-1]["test/acc"],
            "loss": results[-1]["test/loss"],
        }
        return results

    def infer(self, x: Union[Timeseries, Vector, torch.Tensor]) -> Category:
        """Run inference on a data point, x

        Args:
            x (Union[Timeseries, Vector, torch.Tensor])): Either a single time instance (Vector) or a Timeseries.
                x can also be passed as a torch.Tensor.

        Returns:
            Category: Network output
        """

        if isinstance(x, Vector):
            x = torch.tensor(x.data)
            # assert len(data) == self._input_dims
            forward_mode = "step"
            x = x.unsqueeze(0)  # Add batch dim
        elif isinstance(x, Timeseries):
            x = torch.tensor(x.data).permute(1, 0)
            assert x.shape == (self._input_dims, self._sequence_len)
            forward_mode = "regular"
            x = x.unsqueeze(0)  # Add batch dim
        else:
            assert isinstance(x, torch.Tensor)
            if len(x.shape) == 1:
                assert x.shape == (self._input_dims,)
                forward_mode = "step"
                x = x.unsqueeze(0)  # Add batch dim
            elif len(x.shape) == 2:
                if x.shape == (self.batch_size, self._input_dims):
                    forward_mode = "step"
                else:
                    assert x.shape == (self._input_dims, self._sequence_len)
                    forward_mode = "regular"
                    x = x.unsqueeze(0)  # Add batch dim
            else:
                assert len(x.shape) == 3
                assert x.shape == (
                    self.batch_size,
                    self._input_dims,
                    self._sequence_len,
                )
                forward_mode = "regular"

        x = x.to(device=self.device, dtype=torch.float)

        if self._ort_session is not None and self._ort_state is not None and forward_mode == "step":
            inputs = {
                "input": x.cpu().detach().numpy(),
                **self._ort_state,
            }
            r, *next_state = self._ort_session.run(None, inputs)
            r = torch.tensor(r)
            self._ort_state = {k: v for k, v in zip(self._ort_state.keys(), next_state)}
        else:
            self.model.eval()
            r = (self.model.forward if forward_mode == "regular" else self.model.forward_step)(x)
        if isinstance(r, torch.Tensor):
            r = torch.nn.functional.softmax(r[0], dim=-1)
            result = Category(prediction=int(r.argmax(dim=0)), confidence=r)
        else:
            # In the "continual inference"-mode, the model needs to warm up by first seeing
            # "sequence_len" time-steps. Until then, it will output co.TensorPlaceholder values
            result = Category(
                prediction=-1,
                confidence=torch.zeros(self._num_classes, dtype=torch.float),
            )
        return result

    def optimize(self, do_constant_folding=False):
        """Optimize model execution.
        This is accomplished by saving to the ONNX format and loading the optimized model.

        Args:
            do_constant_folding (bool, optional): Whether to optimize constants. Defaults to False.
        """
        if getattr(self.model, "_ort_session", None):
            logger.info("Model is already optimized. Skipping redundant optimization")
            return

        path = Path(self.temp_path or os.getcwd()) / "weights"
        if not path.exists():
            self._save_onnx(path, do_constant_folding)
        self._load_onnx(path)

    @property
    def _example_input(self):
        return torch.randn(1, self._input_dims, self._sequence_len).to(device=self.device)

    @_example_input.setter
    def _example_input(self):
        raise ValueError("_example_input is set thorugh 'sequence_len' and 'input_dims' parameters")

    def _save_onnx(self, path: Union[str, Path], do_constant_folding=False, verbose=False):
        """Save model in the ONNX format

        Args:
            path (Union[str, Path]): Directory in which to save ONNX model
            do_constant_folding (bool, optional): Whether to optimize constants. Defaults to False.
        """
        assert (
            int(torch.__version__.split(".")[1]) >= 10
        ), "ONNX optimization of the Continual Transformer Encoder requires torch >= 1.10.0."

        assert (
            int(getattr(co, "__version__", "0.0.0").split(".")[0]) >= 1
        ), "ONNX optimization of the Continual Transformer Encoder requires continual-inference >= 1.0.0."

        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)

        self.model.eval()
        self.model.to(device="cpu")

        # Prepare state
        state0 = None
        with torch.no_grad():
            for i in range(self._sequence_len):
                _, state0 = self.model._forward_step(self._example_input[:, :, i], state0)
        state0 = co.utils.flatten(state0)

        # Export to ONNX
        onnx_path = path / "cotransenc_weights.onnx"
        logger.info(f"Saving model to ONNX format at {str(onnx_path)}")
        onnx.export(
            self.model,
            (self._example_input[:, :, -1], *state0),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=do_constant_folding,
            verbose=verbose,
            opset_version=14,
        )

        # Save default state and name mappings for later use
        state_path = path.parent / "cotransenc_state.pickle"
        logger.info(f"Saving ONNX model states at {str(state_path)}")
        omodel = onnx.OnnxWrapper(self.model)
        state = {k: v.detach().numpy() for k, v in zip(omodel.state_input_names, state0)}
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

    def _load_onnx(self, path: Union[str, Path]):
        """Loads ONNX model into an onnxruntime inference session.

        Args:
            path (Union[str, Path]): Path to ONNX model folder
        """
        assert (
            int(getattr(ort, "__version__", "0.0.0").split(".")[1]) >= 11
        ), "ONNX inference of the Continual Transformer Encoder requires onnxruntime >= 1.11.0."

        onnx_path = path / "cotransenc_weights.onnx"
        state_path = path.parent / "cotransenc_state.pickle"

        logger.info(f"Loading ONNX runtime inference session from {str(onnx_path)}")
        self._ort_session = ort.InferenceSession(str(onnx_path))

        logger.info(f"Loading ONNX state from {str(state_path)}")
        with open(state_path, "rb") as f:
            self._ort_state = pickle.load(f)


def _experiment_logger():
    return pl.loggers.TensorBoardLogger(save_dir=Path(os.getcwd()) / "logs", name="cotransenc")


def _accuracy(x, y):
    return torch.sum(x.argmax(dim=1) == y) / len(y)


class _LightningModuleWithCrossEntropy(pl.LightningModule):
    def __init__(self, module):
        pl.LightningModule.__init__(self)
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.module(x)
        loss = torch.nn.functional.cross_entropy(z, y)
        self.log("train/loss", loss)
        self.log("train/acc", _accuracy(z, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = torch.nn.functional.cross_entropy(z, y)
        self.log("val/loss", loss)
        self.log("val/acc", _accuracy(z, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = torch.nn.functional.cross_entropy(z, y)
        self.log("test/loss", loss)
        self.log("test/acc", _accuracy(z, y))
        return loss
