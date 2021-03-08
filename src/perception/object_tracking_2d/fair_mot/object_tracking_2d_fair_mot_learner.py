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
from engine.datasets import ExternalDataset, DatasetIterator
from perception.object_tracking_2d.logger import Logger
from perception.object_tracking_2d.datasets.mot_dataset import JointDataset

class ObjectTracking2DFairMotLearner(Learner):
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
        device="cuda:0",
        threshold=0.0,
        scale=1.0,
    ):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(ObjectTracking2DFairMotLearner, self).__init__(
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

        self.__create_model()

    def save(self, path):
        pass

    def load(
        self,
        path,
        silent=False,
        verbose=False,
        logging_path=None,
    ):
        logger = Logger(silent, verbose, logging_path)

        logger.close()

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
        auto_save=False,
        train_split_paths=None,
        val_split_paths=None,
    ):

        if train_split_paths is None:
            train_split_paths = {
                "mot20": "./perception/object_tracking_2d/datasets/data/mot20.train"
            }

        logger = Logger(silent, verbose, logging_path)

        # if model_dir is not None:
        #     model_dir = pathlib.Path(model_dir)
        #     model_dir.mkdir(parents=True, exist_ok=True)
        #     self.model_dir = model_dir

        # if self.model_dir is None and auto_save is True:
        #     raise ValueError(
        #         "Can not use auto_save if model_dir is None and load was not called before"
        #     )

        (
            input_dataset_iterator,
            eval_dataset_iterator,
        ) = self._prepare_datasets(
            dataset,
            val_dataset,
            train_split_paths,
            val_split_paths,
        )

        logger.close()

    def eval(
        self,
        dataset,
        predict_test=False,
        logging_path=None,
        silent=False,
        verbose=False,
    ):

        logger = Logger(silent, verbose, logging_path)

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
        train_split_paths,
        val_split_paths,
        require_dataset=True,
    ):

        input_dataset_iterator = None
        eval_dataset_iterator = None

        if isinstance(dataset, ExternalDataset):

            dataset_path = dataset.path
            if dataset.dataset_type.lower() != "mot":
                raise ValueError(
                    "ExternalDataset (" + str(dataset) +
                    ") is given as a dataset, but it is not a MOT dataset")

            input_dataset_iterator = JointDataset(
                dataset_path,
                train_split_paths,
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
            if val_dataset.dataset_type.lower() != "mot":
                raise ValueError(
                    "ExternalDataset (" + str(val_dataset) +
                    ") is given as a val_dataset, but it is not a MOT dataset"
                )

            eval_dataset_iterator = JointDataset(
                val_dataset_path,
                val_split_paths,
            )

        elif isinstance(val_dataset, DatasetIterator):
            eval_dataset_iterator = dataset
        elif val_dataset is None:
            if isinstance(dataset, ExternalDataset):
                dataset_path = dataset.path
                if dataset.dataset_type.lower() != "mot":
                    raise ValueError(
                        "ExternalDataset (" + str(dataset) +
                        ") is given as a dataset, but it is not a MOT dataset"
                    )

                eval_dataset_iterator = JointDataset(
                    dataset_path,
                    val_split_paths,
                )

            else:
                raise ValueError(
                    "val_dataset is None and can't be derived from" +
                    " the dataset object because the dataset is not an ExternalDataset"
                )
        else:
            raise ValueError(
                "val_dataset parameter should be an ExternalDataset or a DatasetIterator or None"
            )

        return input_dataset_iterator, eval_dataset_iterator

    def __create_model(self):
        pass
