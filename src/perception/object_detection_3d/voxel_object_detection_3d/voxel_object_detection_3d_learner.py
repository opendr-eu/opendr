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

from engine.learners import Learner
from engine.datasets import ExternalDataset, DatasetIterator
from perception.object_detection_3d.voxel_object_detection_3d.model_configs import (
    backbones,
)
from perception.object_detection_3d.voxel_object_detection_3d.second.load import (
    load as second_load,
)
from perception.object_detection_3d.voxel_object_detection_3d.second.run import (
    train,
)
from perception.object_detection_3d.voxel_object_detection_3d.second.pytorch.builder import (
    input_reader_builder,
)


class VoxelObjectDetection3DLearner(Learner):
    # 1. The default values in constructor arguments can be set according to the algorithm.
    # 2. Some of the shared parameters, e.g. lr_schedule, backbone, etc., can be skipped here if not needed
    #    by the implementation.
    # 3. TODO Make sure the naming of the arguments is the same as the parent class arguments to keep it consistent
    #     for the end user.
    def __init__(
        self,
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
        device="cuda",
        threshold=0.0,
        scale=1.0,
        model_config_path=None,
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

        # Define the implementation specific parameters
        # TODO Make sure to do appropriate typechecks and provide valid default values for all custom parameters used.
        # self.model_config = backbones[self.backbone]
        self.model_config_path = model_config_path

    # All methods below are dummy implementations of the abstract methods that are inherited.
    def save(self, path):
        pass

    def load(self, path):
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
            eval_checkpoint_dir,
            float_dtype,
            loss_scale,
            result_path,
            class_names,
            center_limit_range,
        ) = second_load(path, self.model_config_path,)

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
        self.eval_checkpoint_dir = eval_checkpoint_dir
        self.float_dtype = float_dtype
        self.loss_scale = loss_scale
        self.result_path = result_path
        self.class_names = class_names
        self.center_limit_range = center_limit_range

    def optimize(self, params):
        pass

    def reset(self):
        pass

    def fit(
        self,
        dataset,
        refine_weight=2,
        pickle_result=True,
        val_dataset=None,
        logging_path=None,
        silent=False,
        verbose=False,
    ):

        input_dataset_iterator, eval_dataset_iterator = self.__prepare_datasets(
            dataset, val_dataset, self.input_config, self.evaluation_input_config,
            self.model_config, self.voxel_generator, self.target_assigner
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
            self.eval_checkpoint_dir,
            self.float_dtype,
            refine_weight,
            self.loss_scale,
            self.result_path,
            pickle_result,
            self.class_names,
            self.center_limit_range,
            input_dataset_iterator=input_dataset_iterator,
            eval_dataset_iterator=eval_dataset_iterator,
            log_path=logging_path,
            silent=silent,
            verbose=verbose,
        )

    def eval(
        self, dataset, logging_path=None, silent=False, verbose=False,
    ):
        pass

    def __prepare_datasets(
        self,
        dataset,
        val_dataset,
        input_cfg,
        eval_input_cfg,
        model_cfg,
        voxel_generator,
        target_assigner,
    ):

        input_dataset_iterator = None
        eval_dataset_iterator = None

        if isinstance(dataset, ExternalDataset):

            if dataset.dataset_type.lower() != "kitti":
                raise ValueError(
                    "ExternalDataset ("
                    + str(dataset)
                    + ") is given as a dataset, but it is not a KITTI dataset"
                )

            dataset_path = dataset.path
            input_cfg.kitti_info_path = (
                dataset_path + "/" + input_cfg.kitti_info_path
            )
            input_cfg.kitti_root_path = (
                dataset_path + "/" + input_cfg.kitti_root_path
            )
            input_cfg.record_file_path = (
                dataset_path + "/" + input_cfg.record_file_path
            )
            input_cfg.database_sampler.database_info_path = (
                dataset_path
                + "/"
                + input_cfg.database_sampler.database_info_path
            )

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
            raise ValueError(
                "dataset parameter should be an ExternalDataset or a DatasetIterator"
            )

        if isinstance(val_dataset, ExternalDataset):

            val_dataset_path = val_dataset.path
            if val_dataset.dataset_type.lower() != "kitti":
                raise ValueError(
                    "ExternalDataset ("
                    + str(val_dataset)
                    + ") is given as a val_dataset, but it is not a KITTI dataset"
                )

            eval_input_cfg.kitti_info_path = (
                val_dataset_path + "/" + eval_input_cfg.kitti_info_path
            )
            eval_input_cfg.kitti_root_path = (
                val_dataset_path + "/" + eval_input_cfg.kitti_root_path
            )
            eval_input_cfg.record_file_path = (
                val_dataset_path + "/" + eval_input_cfg.record_file_path
            )
            eval_input_cfg.database_sampler.database_info_path = (
                val_dataset_path
                + "/"
                + eval_input_cfg.database_sampler.database_info_path
            )

            eval_dataset_iterator = input_reader_builder.build(
                eval_input_cfg,
                model_cfg,
                training=False,
                voxel_generator=voxel_generator,
                target_assigner=target_assigner,
            )
        elif isinstance(val_dataset, DatasetIterator):
            eval_dataset_iterator = dataset
        elif val_dataset is None:
            if isinstance(dataset, ExternalDataset):
                dataset_path = dataset.path
                if dataset.dataset_type.lower() != "kitti":
                    raise ValueError(
                        "ExternalDataset ("
                        + str(dataset)
                        + ") is given as a dataset, but it is not a KITTI dataset"
                    )

                eval_input_cfg.kitti_info_path = (
                    dataset_path + "/" + eval_input_cfg.kitti_info_path
                )
                eval_input_cfg.kitti_root_path = (
                    dataset_path + "/" + eval_input_cfg.kitti_root_path
                )
                eval_input_cfg.record_file_path = (
                    dataset_path + "/" + eval_input_cfg.record_file_path
                )
                eval_input_cfg.database_sampler.database_info_path = (
                    dataset_path
                    + "/"
                    + eval_input_cfg.database_sampler.database_info_path
                )

                eval_dataset_iterator = input_reader_builder.build(
                    eval_input_cfg,
                    model_cfg,
                    training=False,
                    voxel_generator=voxel_generator,
                    target_assigner=target_assigner,
                )
            else:
                raise ValueError(
                    "val_dataset is None and can't be derived from"
                    + " the dataset object because the dataset is not an ExternalDataset"
                )
        else:
            raise ValueError(
                "dataset parameter should be an ExternalDataset or a DatasetIterator"
            )

        return input_dataset_iterator, eval_dataset_iterator

    def infer(self, batch, tracked_bounding_boxes=None):
        # In this infer dummy implementation, a custom argument is added as optional, so as not to change the basic
        # signature of the abstract method.
        # TODO The implementation must make sure it throws an appropriate error if the custom argument is needed and
        #  not provided (None).
        pass
