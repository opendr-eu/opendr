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

import json
import torch
import yaml
from pathlib import Path
from engine.learners import Learner
from utils.io import bump_version

# from engine.datasets import DatasetIterator, ExternalDataset, MappedDatasetIterator

# from engine.data import Video  # TODO: impl in engine.data
from perception.activity_recognition.x3d.modules.x3d import X3D

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
        lr=0.1,
        iters=10,
        batch_size=64,
        optimizer="sgd",
        device="cuda",
        threshold=0.0,  # Not used
        backbone="s",
        network_head="classification",  # Not used
        temp_path="",
        loss="cross_entropy",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        rgb_mean=None,
        rgb_std=None,
        weight_decay=1e-5,
        momentum=0.9,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        seed=123,
        num_classes=400,
    ):
        assert (
            backbone in _MODEL_NAMES
        ), f"Invalid model selected. Choose one of {_MODEL_NAMES}."
        assert network_head in {
            "classification"
        }, "Currently, only 'classification' head is supported."

        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(X3DLearner, self).__init__(
            lr=lr,
            iters=iters,
            batch_size=batch_size,
            optimizer=optimizer,
            backbone=backbone,
            network_head=network_head,
            temp_path=temp_path,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            device=device,
            threshold=threshold,
        )
        logger.debug("X3DLearner initialising")

        # Defaults as used for pre-trained X3D models
        self.rgb_mean = rgb_mean or (0.45, 0.45, 0.45)
        self.rgb_std = rgb_std or (0.225, 0.225, 0.225)
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
        self.__create_model()
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

    def __create_model(self) -> X3D:
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
                "rgb_mean": self.rgb_mean,
                "rgb_std": self.rgb_std,
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
            rgb_mean=inference_params["rgb_mean"],
            rgb_std=inference_params["rgb_std"],
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

    def fit(
        self,
        dataset,
        val_dataset=None,
        refine_weight=2,
        logging_path=None,
        silent=False,
        verbose=False,
        model_dir=None,
        image_shape=(1224, 370),
        evaluate=True,
    ):
        ...

    def eval(
        self,
        dataset,
        predict_test=False,
        ground_truth_annotations=None,
        logging_path=None,
        silent=False,
        verbose=False,
        image_shape=(1224, 370),
        count=None,
    ):
        ...

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
    #     if self.model is None:
    #         raise UserWarning(
    #             "No model is loaded, cannot optimize. Load or train a model first."
    #         )
    #     if self.model.rpn_ort_session is not None:
    #         raise UserWarning("Model is already optimized in ONNX.")

    #     input_shape = [
    #         1,
    #         self.model.middle_feature_extractor.nchannels,
    #         self.model.middle_feature_extractor.ny,
    #         self.model.middle_feature_extractor.nx,
    #     ]

    #     has_refine = self.model.rpn_class_name in ["PSA", "RefineDet"]

    #     try:
    #         self.__convert_rpn_to_onnx(
    #             input_shape,
    #             has_refine,
    #             os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx"),
    #             do_constant_folding,
    #         )
    #     except FileNotFoundError:
    #         # Create temp directory
    #         os.makedirs(self.temp_path, exist_ok=True)
    #         self.__convert_rpn_to_onnx(
    #             input_shape,
    #             has_refine,
    #             os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx"),
    #             do_constant_folding,
    #         )

    #     self.__load_rpn_from_onnx(
    #         os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx")
    #     )

    # def __convert_rpn_to_onnx(
    #     self,
    #     input_shape,
    #     has_refine,
    #     output_name,
    #     do_constant_folding=False,
    #     verbose=False,
    # ):
    #     inp = torch.randn(input_shape).to(self.device)
    #     input_names = ["data"]
    #     output_names = ["box_preds", "cls_preds", "dir_cls_preds"]

    #     if has_refine:
    #         output_names.append("Refine_loc_preds")
    #         output_names.append("Refine_cls_preds")
    #         output_names.append("Refine_dir_preds")

    #     torch.onnx.export(
    #         self.model.rpn,
    #         inp,
    #         output_name,
    #         verbose=verbose,
    #         enable_onnx_checker=True,
    #         do_constant_folding=do_constant_folding,
    #         input_names=input_names,
    #         output_names=output_names,
    #     )

    # def __load_rpn_from_onnx(self, path):
    #     """
    #     This method loads an ONNX model from the path provided into an onnxruntime inference session.

    #     :param path: path to ONNX model
    #     :type path: str
    #     """
    #     self.model.rpn_ort_session = ort.InferenceSession(path)

    #     # The comments below are the alternative way to use the onnx model, it might be useful in the future
    #     # depending on how ONNX saving/loading will be implemented across the toolkit.
    #     # # Load the ONNX model
    #     # self.model = onnx.load(path)
    #     #
    #     # # Check that the IR is well formed
    #     # onnx.checker.check_model(self.model)
    #     #
    #     # # Print a human readable representation of the graph
    #     # onnx.helper.printable_graph(self.model.graph)

    # def __load_from_pth(self, model, path, use_original_dict=False):
    #     all_params = torch.load(path, map_location=self.device)
    #     model.load_state_dict(
    #         all_params if use_original_dict else all_params["state_dict"]
    #     )

    # def __prepare_datasets(
    #     self,
    #     dataset,
    #     val_dataset,
    #     input_cfg,
    #     eval_input_cfg,
    #     model_cfg,
    #     voxel_generator,
    #     target_assigner,
    #     gt_annos,
    #     require_dataset=True,
    # ):
    #     def create_map_point_cloud_dataset_func(include_annotation_in_example):

    #         prep_func = create_prep_func(
    #             input_cfg,
    #             model_cfg,
    #             True,
    #             voxel_generator,
    #             target_assigner,
    #             use_sampler=False,
    #         )

    #         def map(data):
    #             point_cloud_with_calibration, target = data
    #             point_cloud = point_cloud_with_calibration.data
    #             calib = point_cloud_with_calibration.calib

    #             annotation = target.kitti()

    #             example = _prep_v9(point_cloud, calib, prep_func, annotation)

    #             if include_annotation_in_example:
    #                 example["annos"] = annotation

    #             return example

    #         return map

    #     input_dataset_iterator = None
    #     eval_dataset_iterator = None

    #     if isinstance(dataset, ExternalDataset):

    #         if dataset.dataset_type.lower() != "kitti":
    #             raise ValueError(
    #                 "ExternalDataset ("
    #                 + str(dataset)
    #                 + ") is given as a dataset, but it is not a KITTI dataset"
    #             )

    #         dataset_path = dataset.path
    #         input_cfg.kitti_info_path = dataset_path + "/" + input_cfg.kitti_info_path
    #         input_cfg.kitti_root_path = dataset_path + "/" + input_cfg.kitti_root_path
    #         input_cfg.record_file_path = dataset_path + "/" + input_cfg.record_file_path
    #         input_cfg.database_sampler.database_info_path = (
    #             dataset_path + "/" + input_cfg.database_sampler.database_info_path
    #         )

    #         input_dataset_iterator = input_reader_builder.build(
    #             input_cfg,
    #             model_cfg,
    #             training=True,
    #             voxel_generator=voxel_generator,
    #             target_assigner=target_assigner,
    #         )
    #     elif isinstance(dataset, DatasetIterator):
    #         input_dataset_iterator = MappedDatasetIterator(
    #             dataset, create_map_point_cloud_dataset_func(False),
    #         )
    #     else:
    #         if require_dataset or dataset is not None:
    #             raise ValueError(
    #                 "dataset parameter should be an ExternalDataset or a DatasetIterator"
    #             )

    #     if isinstance(val_dataset, ExternalDataset):

    #         val_dataset_path = val_dataset.path
    #         if val_dataset.dataset_type.lower() != "kitti":
    #             raise ValueError(
    #                 "ExternalDataset ("
    #                 + str(val_dataset)
    #                 + ") is given as a val_dataset, but it is not a KITTI dataset"
    #             )

    #         eval_input_cfg.kitti_info_path = (
    #             val_dataset_path + "/" + eval_input_cfg.kitti_info_path
    #         )
    #         eval_input_cfg.kitti_root_path = (
    #             val_dataset_path + "/" + eval_input_cfg.kitti_root_path
    #         )
    #         eval_input_cfg.record_file_path = (
    #             val_dataset_path + "/" + eval_input_cfg.record_file_path
    #         )
    #         eval_input_cfg.database_sampler.database_info_path = (
    #             val_dataset_path
    #             + "/"
    #             + eval_input_cfg.database_sampler.database_info_path
    #         )

    #         eval_dataset_iterator = input_reader_builder.build(
    #             eval_input_cfg,
    #             model_cfg,
    #             training=False,
    #             voxel_generator=voxel_generator,
    #             target_assigner=target_assigner,
    #         )

    #         if gt_annos is None:
    #             gt_annos = [
    #                 info["annos"] for info in eval_dataset_iterator.dataset.kitti_infos
    #             ]

    #     elif isinstance(val_dataset, DatasetIterator):
    #         eval_dataset_iterator = MappedDatasetIterator(
    #             val_dataset, create_map_point_cloud_dataset_func(True),
    #         )
    #     elif val_dataset is None:
    #         if isinstance(dataset, ExternalDataset):
    #             dataset_path = dataset.path
    #             if dataset.dataset_type.lower() != "kitti":
    #                 raise ValueError(
    #                     "ExternalDataset ("
    #                     + str(dataset)
    #                     + ") is given as a dataset, but it is not a KITTI dataset"
    #                 )

    #             eval_input_cfg.kitti_info_path = (
    #                 dataset_path + "/" + eval_input_cfg.kitti_info_path
    #             )
    #             eval_input_cfg.kitti_root_path = (
    #                 dataset_path + "/" + eval_input_cfg.kitti_root_path
    #             )
    #             eval_input_cfg.record_file_path = (
    #                 dataset_path + "/" + eval_input_cfg.record_file_path
    #             )
    #             eval_input_cfg.database_sampler.database_info_path = (
    #                 dataset_path
    #                 + "/"
    #                 + eval_input_cfg.database_sampler.database_info_path
    #             )

    #             eval_dataset_iterator = input_reader_builder.build(
    #                 eval_input_cfg,
    #                 model_cfg,
    #                 training=False,
    #                 voxel_generator=voxel_generator,
    #                 target_assigner=target_assigner,
    #             )

    #             if gt_annos is None:
    #                 gt_annos = [
    #                     info["annos"]
    #                     for info in eval_dataset_iterator.dataset.kitti_infos
    #                 ]
    #         else:
    #             raise ValueError(
    #                 "val_dataset is None and can't be derived from"
    #                 + " the dataset object because the dataset is not an ExternalDataset"
    #             )
    #     else:
    #         raise ValueError(
    #             "val_dataset parameter should be an ExternalDataset or a DatasetIterator or None"
    #         )

    #     return input_dataset_iterator, eval_dataset_iterator, gt_annos

    # @staticmethod
    # def __extract_trailing(path):
    #     """
    #     Extracts the trailing folder name or filename from a path provided in an OS-generic way, also handling
    #     cases where the last trailing character is a separator. Returns the folder name and the split head and tail.
    #     :param path: the path to extract the trailing filename or folder name from
    #     :type path: str
    #     :return: the folder name, the head and tail of the path
    #     :rtype: tuple of three strings
    #     """
    #     head, tail = ntpath.split(path)
    #     folder_name = tail or ntpath.basename(head)  # handle both a/b/c and a/b/c/
    #     return folder_name, head, tail
