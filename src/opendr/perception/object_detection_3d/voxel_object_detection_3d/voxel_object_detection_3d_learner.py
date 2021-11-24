# Copyright 2020 Aristotle University of Thessaloniki
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

import os
import json
import torch
import ntpath
import shutil
import pathlib
import onnxruntime as ort
from opendr.engine.learners import Learner
from opendr.engine.datasets import DatasetIterator, ExternalDataset, MappedDatasetIterator
from opendr.engine.data import PointCloud
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.load import (
    create_model as second_create_model,
    load_from_checkpoint,
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.run import (
    compute_lidar_kitti_output, evaluate, example_convert_to_torch, train
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.builder import (
    input_reader_builder, )
from opendr.perception.object_detection_3d.voxel_object_detection_3d.logger import (
    Logger, )
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.models.tanet import (
    set_tanet_config
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.data.preprocess import (
    _prep_v9, _prep_v9_infer
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.builder.dataset_builder import (
    create_prep_func
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.data.preprocess import (
    merge_second_batch,
)
from opendr.engine.target import BoundingBox3DList
from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve
from urllib.error import URLError
import warnings
from numba import errors

original_warn = warnings.warn


def warn(warning, *args, **kwargs):
    if not isinstance(warning, errors.NumbaPerformanceWarning):
        original_warn(warning, *args, **kwargs)


warnings.warn = warn


class VoxelObjectDetection3DLearner(Learner):
    def __init__(
        self,
        model_config_path,
        lr=0.0002,
        iters=10,
        batch_size=64,
        optimizer="adam_optimizer",
        lr_schedule="exponential_decay_learning_rate",
        backbone="tanet_16",
        network_head="",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="",
        device="cuda:0",
        threshold=0.0,
        scale=1.0,
        tanet_config_path=None,
        optimizer_params={
            "weight_decay": 0.0001,
        },
        lr_schedule_params={
            "decay_steps": 27840,
            "decay_factor": 0.8,
            "staircase": True,
        }
    ):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(VoxelObjectDetection3DLearner, self).__init__(
            lr=lr,
            iters=iters,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            backbone=backbone,
            network_head=network_head,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            temp_path=temp_path,
            device=device,
            threshold=threshold,
            scale=scale,
        )

        self.model_config_path = model_config_path
        self.optimizer_params = optimizer_params
        self.lr_schedule_params = lr_schedule_params

        self.model_dir = None
        self.eval_checkpoint_dir = None
        self.infer_point_cloud_mapper = None

        if tanet_config_path is not None:
            set_tanet_config(tanet_config_path)

        self.__create_model()

        self.model.rpn_ort_session = None  # ONNX runtime inference session

    def save(self, path, verbose=False):
        """
        This method is used to save a trained model.
        Provided with the path, absolute or relative, including a *folder* name, it creates a directory with the name
        of the *folder* provided and saves the model inside with a proper format and a .json file with metadata.
        If self.optimize was ran previously, it saves the optimized ONNX model in a similar fashion, by copying it
        from the self.temp_path it was saved previously during conversion.
        :param path: for the model to be saved, including the folder name
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """

        if self.model is None:
            raise UserWarning("No model is loaded, cannot save.")

        folder_name, _, tail = self.__extract_trailing(path)  # Extract trailing folder name from path
        # Also extract folder name without any extension if extension is erroneously provided
        folder_name_no_ext = folder_name.split(sep='.')[0]

        # Extract path without folder name, by removing folder name from original path
        path_no_folder_name = ''.join(path.rsplit(folder_name, 1))
        # If tail is '', then path was a/b/c/, which leaves a trailing double '/'
        if tail == '':
            path_no_folder_name = path_no_folder_name[0:-1]  # Remove one '/'

        # Create model directory
        new_path = path_no_folder_name + folder_name_no_ext
        os.makedirs(new_path, exist_ok=True)

        model_metadata = {"model_paths": [], "framework": "pytorch", "format": "", "has_data": False,
                          "inference_params": {}, "optimized": None, "optimizer_info": {}}

        if self.model.rpn_ort_session is None:
            model_metadata["model_paths"] = [
                folder_name_no_ext + "_vfe.pth",
                folder_name_no_ext + "_mfe.pth",
                folder_name_no_ext + "_rpn.pth"
            ]
            model_metadata["optimized"] = False
            model_metadata["format"] = "pth"

            torch.save({
                'state_dict': self.model.voxel_feature_extractor.state_dict()
            }, os.path.join(path_no_folder_name, folder_name_no_ext, model_metadata["model_paths"][0]))
            torch.save({
                'state_dict': self.model.middle_feature_extractor.state_dict()
            }, os.path.join(path_no_folder_name, folder_name_no_ext, model_metadata["model_paths"][1]))
            torch.save({
                'state_dict': self.model.rpn.state_dict()
            }, os.path.join(path_no_folder_name, folder_name_no_ext, model_metadata["model_paths"][2]))
            if verbose:
                print("Saved Pytorch VFE, MFE and RPN sub-models.")
        else:
            model_metadata["model_paths"] = [
                folder_name_no_ext + "_vfe.pth",
                folder_name_no_ext + "_mfe.pth",
                folder_name_no_ext + "_rpn.onnx"
            ]
            model_metadata["optimized"] = True
            model_metadata["format"] = "onnx"

            torch.save({
                'state_dict': self.model.voxel_feature_extractor.state_dict()
            }, os.path.join(path_no_folder_name, folder_name_no_ext, model_metadata["model_paths"][0]))
            torch.save({
                'state_dict': self.model.middle_feature_extractor.state_dict()
            }, os.path.join(path_no_folder_name, folder_name_no_ext, model_metadata["model_paths"][1]))
            # Copy already optimized model from temp path
            shutil.copy2(
                os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx"),
                os.path.join(path_no_folder_name, folder_name_no_ext, model_metadata["model_paths"][2])
            )
            if verbose:
                print("Saved Pytorch VFE, MFE and ONNX RPN sub-models.")

        with open(os.path.join(new_path, folder_name_no_ext + ".json"), 'w') as outfile:
            json.dump(model_metadata, outfile)

    def load(
        self,
        path,
        verbose=False,
    ):
        """
        Loads the model from inside the path provided, based on the metadata .json file included.
        :param path: path of the directory the model was saved
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """

        model_name, _, _ = self.__extract_trailing(path)  # Trailing folder name from the path provided

        with open(os.path.join(path, model_name + ".json")) as metadata_file:
            metadata = json.load(metadata_file)

        if len(metadata["model_paths"]) == 1:
            self.__load_from_pth(self.model, os.path.join(path, metadata["model_paths"][0]), True)
            if verbose:
                print("Loaded Pytorch model.")
        else:
            self.__load_from_pth(self.model.voxel_feature_extractor, os.path.join(path, metadata["model_paths"][0]))
            self.__load_from_pth(self.model.middle_feature_extractor, os.path.join(path, metadata["model_paths"][1]))
            if verbose:
                print("Loaded Pytorch VFE and MFE sub-model.")

            if not metadata["optimized"]:
                self.__load_from_pth(self.model.rpn, os.path.join(path, metadata["model_paths"][2]))
                if verbose:
                    print("Loaded Pytorch RPN sub-model.")
            else:
                self.__load_rpn_from_onnx(os.path.join(path, metadata["model_paths"][2]))
                if verbose:
                    print("Loaded ONNX RPN sub-model.")

    def reset(self):
        pass

    def fit(
        self,
        dataset,
        val_dataset=None,
        refine_weight=2,
        ground_truth_annotations=None,
        logging_path=None,
        silent=False,
        verbose=False,
        model_dir=None,
        image_shape=(1224, 370),
        evaluate=True,
    ):

        logger = Logger(silent, verbose, logging_path)
        display_step = 1 if verbose else 50

        if model_dir is not None:
            model_dir = pathlib.Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir = model_dir

        if self.model_dir is None and (
            self.checkpoint_load_iter != 0 or
            self.checkpoint_after_iter != 0
        ):
            raise ValueError(
                "Can not use checkpoint_load_iter or checkpoint_after_iter if model_dir is None and load was not called before"
            )

        (
            input_dataset_iterator,
            eval_dataset_iterator,
            ground_truth_annotations,
        ) = self.__prepare_datasets(
            dataset,
            val_dataset,
            self.input_config,
            self.evaluation_input_config,
            self.model_config,
            self.voxel_generator,
            self.target_assigner,
            ground_truth_annotations,
        )

        checkpoints_path = self.model_dir / "checkpoints"
        if self.checkpoint_after_iter != 0 or self.checkpoint_load_iter != 0:
            checkpoints_path.mkdir(exist_ok=True)

        if self.checkpoint_load_iter != 0:
            self.lr_scheduler = load_from_checkpoint(
                self.model, self.mixed_optimizer,
                checkpoints_path / f"checkpoint_{self.checkpoint_load_iter}.pth",
                self.lr_schedule, self.lr_schedule_params, self.device
            )

        train(
            self.model,
            self.input_config,
            self.train_config,
            self.evaluation_input_config,
            self.model_config,
            self.mixed_optimizer,
            self.lr_scheduler,
            self.model_dir,
            self.float_dtype,
            refine_weight,
            self.loss_scale,
            self.class_names,
            self.center_limit_range,
            input_dataset_iterator=input_dataset_iterator,
            eval_dataset_iterator=eval_dataset_iterator,
            gt_annos=ground_truth_annotations,
            log=logger.log,
            checkpoint_after_iter=self.checkpoint_after_iter,
            checkpoints_path=checkpoints_path,
            display_step=display_step,
            device=self.device,
            image_shape=image_shape,
            evaluate=evaluate,
        )

        logger.close()

    def eval(
        self,
        dataset,
        ground_truth_annotations=None,
        logging_path=None,
        silent=False,
        verbose=False,
        image_shape=(370, 1224),
        count=None,
    ):

        logger = Logger(silent, verbose, logging_path)

        (
            _,
            eval_dataset_iterator,
            ground_truth_annotations,
        ) = self.__prepare_datasets(
            None,
            dataset,
            self.input_config,
            self.evaluation_input_config,
            self.model_config,
            self.voxel_generator,
            self.target_assigner,
            ground_truth_annotations,
            require_dataset=False,
        )

        result = evaluate(
            self.model,
            self.evaluation_input_config,
            self.model_config,
            self.mixed_optimizer,
            self.model_dir,
            self.float_dtype,
            self.class_names,
            self.center_limit_range,
            eval_dataset_iterator=eval_dataset_iterator,
            gt_annos=ground_truth_annotations,
            predict_test=False,
            log=logger.log,
            device=self.device,
            image_shape=image_shape,
            count=count,
        )

        logger.close()

        return result

    def infer(self, point_clouds):

        if self.model is None:
            raise ValueError("No model loaded or created")

        if self.infer_point_cloud_mapper is None:
            prep_func = create_prep_func(
                self.input_config,
                self.model_config,
                False,
                self.voxel_generator,
                self.target_assigner,
                use_sampler=False,
            )

            def infer_point_cloud_mapper(x):
                return _prep_v9_infer(x, prep_func)

            self.infer_point_cloud_mapper = infer_point_cloud_mapper
            self.model.eval()

        input_data = None

        if isinstance(point_clouds, PointCloud):
            input_data = merge_second_batch(
                [self.infer_point_cloud_mapper(point_clouds.data)]
            )
        elif isinstance(point_clouds, list):
            input_data = merge_second_batch(
                [self.infer_point_cloud_mapper(x.data) for x in point_clouds]
            )
        else:
            return ValueError(
                "point_clouds should be a PointCloud or a list of PointCloud"
            )

        output = self.model(example_convert_to_torch(
            input_data,
            self.float_dtype,
            device=self.device,
        ))

        if (
            self.model_config.rpn.module_class_name == "PSA" or
            self.model_config.rpn.module_class_name == "RefineDet"
        ):
            output = output[-1]

        annotations = compute_lidar_kitti_output(
            output, self.center_limit_range, self.class_names, None)

        result = [BoundingBox3DList.from_kitti(anno) for anno in annotations]

        if isinstance(point_clouds, PointCloud):
            return result[0]

        return result

    def optimize(self, do_constant_folding=False):
        """
        Optimize method converts the model to ONNX format and saves the
        model in the parent directory defined by self.temp_path. The ONNX model is then loaded.
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        if self.model is None:
            raise UserWarning("No model is loaded, cannot optimize. Load or train a model first.")
        if self.model.rpn_ort_session is not None:
            raise UserWarning("Model is already optimized in ONNX.")

        input_shape = [
            1,
            self.model.middle_feature_extractor.nchannels,
            self.model.middle_feature_extractor.ny,
            self.model.middle_feature_extractor.nx,
        ]

        has_refine = self.model.rpn_class_name in ["PSA", "RefineDet"]

        try:
            self.__convert_rpn_to_onnx(
                input_shape, has_refine,
                os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx"), do_constant_folding
            )
        except FileNotFoundError:
            # Create temp directory
            os.makedirs(self.temp_path, exist_ok=True)
            self.__convert_rpn_to_onnx(
                input_shape, has_refine,
                os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx"), do_constant_folding
            )

        self.__load_rpn_from_onnx(os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx"))

    @staticmethod
    def download(model_name, path, server_url=None):

        if server_url is not None and model_name not in [
            "pointpillars_car_xyres_16",
            "pointpillars_ped_cycle_xyres_16",
            "tanet_car_xyres_16",
            "tanet_ped_cycle_xyres_16",
        ]:
            raise ValueError("Unknown model_name: " + model_name)

        os.makedirs(path, exist_ok=True)

        if server_url is None:
            server_url = os.path.join(
                OPENDR_SERVER_URL, "perception", "object_detection_3d",
                "voxel_object_detection_3d"
            )

        url = os.path.join(
            server_url, model_name
        )

        model_dir = os.path.join(path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        urlretrieve(os.path.join(
            url, model_name + ".json"
        ), os.path.join(
            model_dir, model_name + ".json"
        ))

        try:
            urlretrieve(os.path.join(
                url, model_name + ".pth"
            ), os.path.join(
                model_dir, model_name + ".pth"
            ))
        except URLError:
            urlretrieve(os.path.join(
                url, model_name + ".tckpt"
            ), os.path.join(
                model_dir, model_name + ".pth"
            ))

        print("Downloaded model", model_name, "to", model_dir)

        return model_dir

    def __convert_rpn_to_onnx(self, input_shape, has_refine, output_name, do_constant_folding=False, verbose=False):
        inp = torch.randn(input_shape).to(self.device)
        input_names = ["data"]
        output_names = [
            "box_preds", "cls_preds", "dir_cls_preds"
        ]

        if has_refine:
            output_names.append("Refine_loc_preds")
            output_names.append("Refine_cls_preds")
            output_names.append("Refine_dir_preds")

        torch.onnx.export(
            self.model.rpn, inp, output_name, verbose=verbose, enable_onnx_checker=True,
            do_constant_folding=do_constant_folding, input_names=input_names, output_names=output_names
        )

    def __load_rpn_from_onnx(self, path):
        """
        This method loads an ONNX model from the path provided into an onnxruntime inference session.

        :param path: path to ONNX model
        :type path: str
        """
        self.model.rpn_ort_session = ort.InferenceSession(path)

        # The comments below are the alternative way to use the onnx model, it might be useful in the future
        # depending on how ONNX saving/loading will be implemented across the toolkit.
        # # Load the ONNX model
        # self.model = onnx.load(path)
        #
        # # Check that the IR is well formed
        # onnx.checker.check_model(self.model)
        #
        # # Print a human readable representation of the graph
        # onnx.helper.printable_graph(self.model.graph)

    def __load_from_pth(self, model, path, use_original_dict=False):
        all_params = torch.load(path, map_location=self.device)
        model.load_state_dict(all_params if use_original_dict else all_params["state_dict"])

    def __prepare_datasets(
        self,
        dataset,
        val_dataset,
        input_cfg,
        eval_input_cfg,
        model_cfg,
        voxel_generator,
        target_assigner,
        gt_annos,
        require_dataset=True,
    ):

        def create_map_point_cloud_dataset_func(is_training):

            prep_func = create_prep_func(
                input_cfg if is_training else eval_input_cfg,
                model_cfg, is_training,
                voxel_generator, target_assigner,
                use_sampler=False,
            )

            def map(data_target):

                point_cloud_with_calibration, target = data_target
                point_cloud = point_cloud_with_calibration.data
                calib = point_cloud_with_calibration.calib

                annotation = target.kitti()

                example = _prep_v9(point_cloud, calib, prep_func, annotation)

                if not is_training:
                    example["annos"] = annotation

                if point_cloud_with_calibration.image_shape is not None:
                    example["image_shape"] = point_cloud_with_calibration.image_shape

                return example

            return map

        input_dataset_iterator = None
        eval_dataset_iterator = None

        if isinstance(dataset, ExternalDataset):

            if dataset.dataset_type.lower() != "kitti":
                raise ValueError(
                    "ExternalDataset (" + str(dataset) +
                    ") is given as a dataset, but it is not a KITTI dataset")

            dataset_path = dataset.path
            input_cfg.kitti_info_path = (dataset_path + "/" +
                                         input_cfg.kitti_info_path)
            input_cfg.kitti_root_path = (dataset_path + "/" +
                                         input_cfg.kitti_root_path)
            input_cfg.record_file_path = (dataset_path + "/" +
                                          input_cfg.record_file_path)
            input_cfg.database_sampler.database_info_path = (
                dataset_path + "/" +
                input_cfg.database_sampler.database_info_path)

            input_dataset_iterator = input_reader_builder.build(
                input_cfg,
                model_cfg,
                training=True,
                voxel_generator=voxel_generator,
                target_assigner=target_assigner,
            )
        elif isinstance(dataset, DatasetIterator):
            input_dataset_iterator = MappedDatasetIterator(
                dataset,
                create_map_point_cloud_dataset_func(True),
            )
        else:
            if require_dataset or dataset is not None:
                raise ValueError(
                    "dataset parameter should be an ExternalDataset or a DatasetIterator"
                )

        if isinstance(val_dataset, ExternalDataset):

            val_dataset_path = val_dataset.path
            if val_dataset.dataset_type.lower() != "kitti":
                raise ValueError(
                    "ExternalDataset (" + str(val_dataset) +
                    ") is given as a val_dataset, but it is not a KITTI dataset"
                )

            eval_input_cfg.kitti_info_path = (val_dataset_path + "/" +
                                              eval_input_cfg.kitti_info_path)
            eval_input_cfg.kitti_root_path = (val_dataset_path + "/" +
                                              eval_input_cfg.kitti_root_path)
            eval_input_cfg.record_file_path = (val_dataset_path + "/" +
                                               eval_input_cfg.record_file_path)
            eval_input_cfg.database_sampler.database_info_path = (
                val_dataset_path + "/" +
                eval_input_cfg.database_sampler.database_info_path)

            eval_dataset_iterator = input_reader_builder.build(
                eval_input_cfg,
                model_cfg,
                training=False,
                voxel_generator=voxel_generator,
                target_assigner=target_assigner,
            )

            if gt_annos is None:
                gt_annos = [
                    info["annos"]
                    for info in eval_dataset_iterator.dataset.kitti_infos
                ]

        elif isinstance(val_dataset, DatasetIterator):
            eval_dataset_iterator = MappedDatasetIterator(
                val_dataset,
                create_map_point_cloud_dataset_func(False),
            )
        elif val_dataset is None:
            if isinstance(dataset, ExternalDataset):
                dataset_path = dataset.path
                if dataset.dataset_type.lower() != "kitti":
                    raise ValueError(
                        "ExternalDataset (" + str(dataset) +
                        ") is given as a dataset, but it is not a KITTI dataset"
                    )

                eval_input_cfg.kitti_info_path = (
                    dataset_path + "/" + eval_input_cfg.kitti_info_path)
                eval_input_cfg.kitti_root_path = (
                    dataset_path + "/" + eval_input_cfg.kitti_root_path)
                eval_input_cfg.record_file_path = (
                    dataset_path + "/" + eval_input_cfg.record_file_path)
                eval_input_cfg.database_sampler.database_info_path = (
                    dataset_path + "/" +
                    eval_input_cfg.database_sampler.database_info_path)

                eval_dataset_iterator = input_reader_builder.build(
                    eval_input_cfg,
                    model_cfg,
                    training=False,
                    voxel_generator=voxel_generator,
                    target_assigner=target_assigner,
                )

                if gt_annos is None:
                    gt_annos = [
                        info["annos"]
                        for info in eval_dataset_iterator.dataset.kitti_infos
                    ]
            else:
                raise ValueError(
                    "val_dataset is None and can't be derived from" +
                    " the dataset object because the dataset is not an ExternalDataset"
                )
        else:
            raise ValueError(
                "val_dataset parameter should be an ExternalDataset or a DatasetIterator or None"
            )

        return input_dataset_iterator, eval_dataset_iterator, gt_annos

    def __create_model(self):
        (
            model,
            input_config,
            train_config,
            evaluation_input_config,
            model_config,
            voxel_generator,
            target_assigner,
            mixed_optimizer,
            lr_scheduler,
            float_dtype,
            loss_scale,
            class_names,
            center_limit_range,
        ) = second_create_model(
            self.model_config_path, device=self.device,
            optimizer_name=self.optimizer,
            optimizer_params=self.optimizer_params,
            lr=self.lr,
            lr_schedule_name=self.lr_schedule,
            lr_schedule_params=self.lr_schedule_params,
        )

        self.model = model
        self.input_config = input_config
        self.train_config = train_config
        self.evaluation_input_config = evaluation_input_config
        self.model_config = model_config
        self.train_config = train_config
        self.voxel_generator = voxel_generator
        self.target_assigner = target_assigner
        self.mixed_optimizer = mixed_optimizer
        self.lr_scheduler = lr_scheduler

        self.float_dtype = float_dtype
        self.loss_scale = loss_scale
        self.class_names = class_names
        self.center_limit_range = center_limit_range

    @staticmethod
    def __extract_trailing(path):
        """
        Extracts the trailing folder name or filename from a path provided in an OS-generic way, also handling
        cases where the last trailing character is a separator. Returns the folder name and the split head and tail.
        :param path: the path to extract the trailing filename or folder name from
        :type path: str
        :return: the folder name, the head and tail of the path
        :rtype: tuple of three strings
        """
        head, tail = ntpath.split(path)
        folder_name = tail or ntpath.basename(head)  # handle both a/b/c and a/b/c/
        return folder_name, head, tail
