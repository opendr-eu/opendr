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

import torch
import torchplus
import pathlib
from engine.learners import Learner
from engine.datasets import ExternalDataset, DatasetIterator
from perception.object_detection_3d.voxel_object_detection_3d.second.load import (
    load as second_load,
    create_model as second_create_model,
)
from perception.object_detection_3d.voxel_object_detection_3d.second.run import (
    train,
    evaluate,
)
from perception.object_detection_3d.voxel_object_detection_3d.second.pytorch.builder import (
    input_reader_builder, )
from perception.object_detection_3d.voxel_object_detection_3d.logger import (
    Logger, )
from perception.object_detection_3d.voxel_object_detection_3d.second.pytorch.models.tanet import set_tanet_config


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
        )

        logger.close()

        return result

    def infer(self, batch):

        if self.model is None:
            raise ValueError("No model loaded or created")

        self.model.eval()

        result = self.model(batch)

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
            input_dataset_iterator = dataset
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
            eval_dataset_iterator = dataset
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
