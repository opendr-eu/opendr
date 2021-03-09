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

import pathlib
from engine.learners import Learner
from engine.datasets import DatasetIterator, ExternalDataset, MappedDatasetIterator
from engine.data import PointCloud
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.load import (
    load as second_load,
    create_model as second_create_model,
)
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.run import (
    compute_lidar_kitti_output, evaluate, example_convert_to_torch, train
)
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.builder import (
    input_reader_builder, )
from perception.object_detection_3d.voxel_object_detection_3d.logger import (
    Logger, )
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.models.tanet import set_tanet_config
import torchplus
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.data.preprocess import _prep_v9, _prep_v9_infer
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.builder.dataset_builder import create_prep_func
from perception.object_detection_3d.voxel_object_detection_3d.second_detector.data.preprocess import (
    merge_second_batch,
)
from engine.target import BoundingBox3DList


class VoxelObjectDetection3DLearner(Learner):
    def __init__(
        self,
        model_config_path,
        lr=0.001,
        iters=10,
        batch_size=64,
        optimizer="sgd",
        lr_schedule="",
        backbone="tanet_16",
        network_head="",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="",
        device="cuda:0",
        threshold=0.0,
        scale=1.0,
        tanet_config_path=None,
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
        self.model_dir = None
        self.eval_checkpoint_dir = None
        self.result_path = None
        self.infer_point_cloud_mapper = None

        if tanet_config_path is not None:
            set_tanet_config(tanet_config_path)

        self.__create_model()

    def save(self, path, max_to_keep=100):

        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torchplus.train.save_models(
            path,
            [self.model, self.mixed_optimizer],
            self.model.get_global_step(),
            max_to_keep=max_to_keep,
        )

        if self.model_dir is None:
            self.model_dir = path

    def load(
        self,
        path,
        silent=False,
        verbose=False,
        logging_path=None,
    ):
        logger = Logger(silent, verbose, logging_path)

        (
            model,
            input_config,
            train_config,
            evaluation_input_config,
            model_config,
            train_config,
            voxel_generator,
            target_assigner,
            mixed_optimizer,
            lr_scheduler,
            model_dir,
            float_dtype,
            loss_scale,
            result_path,
            class_names,
            center_limit_range,
        ) = second_load(
            path,
            self.model_config_path,
            device=self.device,
            log=lambda *x: logger.log(Logger.LOG_WHEN_VERBOSE, *x),
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

        self.model_dir = model_dir
        self.float_dtype = float_dtype
        self.loss_scale = loss_scale
        self.result_path = result_path
        self.class_names = class_names
        self.center_limit_range = center_limit_range

        logger.close()

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
        auto_save=False,
        image_shape=(1224, 370),
    ):

        logger = Logger(silent, verbose, logging_path)
        display_step = 1 if verbose else 50

        if model_dir is not None:
            model_dir = pathlib.Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir = model_dir

        if self.model_dir is None and auto_save is True:
            raise ValueError(
                "Can not use auto_save if model_dir is None and load was not called before"
            )

        (
            input_dataset_iterator,
            eval_dataset_iterator,
            ground_truth_annotations,
        ) = self._prepare_datasets(
            dataset,
            val_dataset,
            self.input_config,
            self.evaluation_input_config,
            self.model_config,
            self.voxel_generator,
            self.target_assigner,
            ground_truth_annotations,
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
            auto_save=auto_save,
            display_step=display_step,
            device=self.device,
            image_shape=image_shape,
        )

        logger.close()

    def eval(
        self,
        dataset,
        predict_test=False,
        ground_truth_annotations=None,
        logging_path=None,
        silent=False,
        verbose=False,
        image_shape=(1224, 370),
    ):

        logger = Logger(silent, verbose, logging_path)

        (
            _,
            eval_dataset_iterator,
            ground_truth_annotations,
        ) = self._prepare_datasets(
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
            predict_test=predict_test,
            log=logger.log,
            device=self.device,
            image_shape=image_shape,
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
        pass

    def _prepare_datasets(
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

        def create_map_point_cloud_dataset_func(include_annotation_in_example):

            prep_func = create_prep_func(
                input_cfg, model_cfg, True,
                voxel_generator, target_assigner,
                use_sampler=False,
            )

            def map(data):
                point_cloud_with_calibration, target = data
                point_cloud = point_cloud_with_calibration.data
                calib = point_cloud_with_calibration.calib

                annotation = target.kitti()

                example = _prep_v9(point_cloud, calib, prep_func, annotation)

                if include_annotation_in_example:
                    example["annos"] = annotation

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
                create_map_point_cloud_dataset_func(False),
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
                create_map_point_cloud_dataset_func(True),
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
            train_config,
            voxel_generator,
            target_assigner,
            mixed_optimizer,
            lr_scheduler,
            float_dtype,
            loss_scale,
            class_names,
            center_limit_range,
        ) = second_create_model(self.model_config_path, device=self.device)

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
