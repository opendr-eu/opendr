# Copyright 2020-2021 OpenDR European Project
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

from perception.object_tracking_2d.datasets.mot_dataset import (
    JointDataset,
    MotDataset,
    MotDatasetIterator,
    RawMotDatasetIterator,
)
from perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import (
    ObjectTracking2DFairMotLearner,
)
from torchvision.transforms import transforms as T
import os


def test_dataset():

    dataset_path = "./perception/object_tracking_2d/datasets/data"
    train_split_paths = {
        # "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train",
        "nano_mot20": "./perception/object_tracking_2d/datasets/splits/nano_mot20.train",
        # "mot17": "./perception/object_tracking_2d/datasets/splits/mot17.train",
    }

    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(
        MotDataset(dataset_path).path,
        train_split_paths,
        augment=True,
        transforms=transforms,
    )

    values = [dataset[2], dataset[1]]

    print(values)
    print()


def test_dataset2():

    train_split_paths = {
        "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train"
    }
    dataset_path = "./perception/object_tracking_2d/datasets/data"

    dataset = MotDatasetIterator(dataset_path, train_split_paths,)

    values = [dataset[2], dataset[1]]

    print(values)
    print()


def v():

    dataset_path = "./perception/object_tracking_2d/datasets/data"
    train_split_paths = {
        # "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train"
        "nano_mot20": "./perception/object_tracking_2d/datasets/splits/nano_mot20.train",
    }

    dataset = MotDatasetIterator(dataset_path, train_split_paths)
    eval_dataset = RawMotDatasetIterator(dataset_path, train_split_paths)

    learner = ObjectTracking2DFairMotLearner(
        iters=3,
        num_epochs=1,
        checkpoint_after_iter=3,
    )

    # starting_param = list(learner.model.parameters())[0].clone()

    learner.fit(
        dataset,
        val_dataset=eval_dataset,
        val_epochs=1,
        train_split_paths=train_split_paths,
        val_split_paths=train_split_paths,
        verbose=True,
    )

    print()

    # learner = ObjectTracking2DFairMotLearner(
    #     iters=3,
    #     num_epochs=10,
    #     checkpoint_after_iter=3,
    #     checkpoint_load_iter=9,
    #     temp_path=os.path.join(
    #         "tests",
    #         "sources",
    #         "tools",
    #         "perception",
    #         "object_tracking_2d",
    #         "fair_mot",
    #         "fair_mot_temp",
    #     ),
    # )

    # learner.fit(
    #     dataset, val_epochs=2,
    #     train_split_paths=train_split_paths,
    #     val_split_paths=train_split_paths,
    #     verbose=True,
    #     # silent=True,
    # )

    # print(learner)
    # print()


def test_fit2():

    train_split_paths = {
        "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train"
    }
    dataset_path = "./perception/object_tracking_2d/datasets/data"

    dataset = MotDatasetIterator(dataset_path, train_split_paths,)

    learner = ObjectTracking2DFairMotLearner(
        iters=10,
        num_epochs=2,
        checkpoint_after_iter=3,
        checkpoint_load_iter=9,
        temp_path=os.path.join(
            "tests",
            "sources",
            "tools",
            "perception",
            "object_tracking_2d",
            "fair_mot",
            "fair_mot_temp",
        ),
    )

    learner.fit(
        dataset,
        val_dataset=dataset,
        val_epochs=1,
        train_split_paths=train_split_paths,
        val_split_paths=train_split_paths,
        verbose=True,
        # silent=True,
    )

    print(learner)
    print()


def test_infer():

    train_split_paths = {
        "nano_mot20": "./perception/object_tracking_2d/datasets/splits/nano_mot20.train",
    }
    dataset_path = "./perception/object_tracking_2d/datasets/data"

    train_dataset = MotDatasetIterator(dataset_path, train_split_paths,)
    infer_dataset = RawMotDatasetIterator(dataset_path, train_split_paths,)

    learner = ObjectTracking2DFairMotLearner(
        iters=3,
        num_epochs=10,
        checkpoint_after_iter=3,
        checkpoint_load_iter=9,
        batch_size=1,
        temp_path=os.path.join(
            "tests",
            "sources",
            "tools",
            "perception",
            "object_tracking_2d",
            "fair_mot",
            "fair_mot_temp",
        ),
    )

    learner.fit(train_dataset)

    val = learner.infer([
        infer_dataset[0][0],
        infer_dataset[1][0],
        infer_dataset[2][0],
    ], [10, 11, 12])

    print(val)
    print()


def test_eval():

    train_split_paths = {
        "nano_mot20": "./perception/object_tracking_2d/datasets/splits/nano_mot20.train",
    }
    dataset_path = "./perception/object_tracking_2d/datasets/data"

    train_dataset = MotDatasetIterator(dataset_path, train_split_paths,)
    eval_dataset = RawMotDatasetIterator(dataset_path, train_split_paths,)

    learner = ObjectTracking2DFairMotLearner(
        iters=3,
        num_epochs=4,
        checkpoint_after_iter=13,
        checkpoint_load_iter=9,
        batch_size=1,
        temp_path=os.path.join(
            "tests",
            "sources",
            "tools",
            "perception",
            "object_tracking_2d",
            "fair_mot",
            "fair_mot_temp",
        ),
    )

    learner.fit(train_dataset)

    val = learner.eval(
        eval_dataset,
    )

    print(val)
    print()


def test_optimize():

    # dataset_path = "./perception/object_tracking_2d/datasets/data"
    # train_split_paths = {
    #     "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train"
    # }

    # dataset = MotDataset(dataset_path)

    learner = ObjectTracking2DFairMotLearner(
        iters=10,
        num_epochs=2,
        checkpoint_after_iter=3,
        checkpoint_load_iter=9,
        temp_path=os.path.join(
            "tests",
            "sources",
            "tools",
            "perception",
            "object_tracking_2d",
            "fair_mot",
            "fair_mot_temp",
        ),
    )

    learner.optimize()

    print(learner)
    print()


def test_save():

    # dataset_path = "./perception/object_tracking_2d/datasets/data"
    # train_split_paths = {
    #     "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train"
    # }

    # dataset = MotDataset(dataset_path)

    learner = ObjectTracking2DFairMotLearner(
        iters=10,
        num_epochs=2,
        checkpoint_after_iter=3,
        checkpoint_load_iter=9,
        temp_path=os.path.join(
            "tests",
            "sources",
            "tools",
            "perception",
            "object_tracking_2d",
            "fair_mot",
            "fair_mot_temp",
        ),
    )
    learner.save("tests")
    learner.load("tests", verbose=True)

    learner.optimize()

    print(learner)
    print()


# test_dataset()
# test_dataset2()
test_eval()
# test_fit()
# test_fit2()
# test_optimize()
# test_save()
