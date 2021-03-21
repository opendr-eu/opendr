from perception.object_tracking_2d.datasets.mot_dataset import JointDataset, MotDataset
from perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import (
    ObjectTracking2DFairMotLearner,
)
from torchvision.transforms import transforms as T
import os


def test_dataset():

    dataset_path = "./perception/object_tracking_2d/datasets/data"
    train_split_paths = {
        "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train",
        # "mot17": "./perception/object_tracking_2d/datasets/splits/mot17.train",
    }

    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(dataset_path, train_split_paths, augment=True, transforms=transforms)

    values = [dataset[2], dataset[1]]

    print(values)
    print()


def test_fit():

    dataset_path = "./perception/object_tracking_2d/datasets/data"
    train_split_paths = {
        "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train"
    }

    dataset = MotDataset(dataset_path)

    learner = ObjectTracking2DFairMotLearner(
        iters=10,
        num_epochs=2,
        checkpoint_after_iter=3,
        checkpoint_load_iter=9,
        temp_path=os.path.join(
            "tests", "sources", "tools",
            "perception", "object_tracking_2d",
            "fair_mot",
            "fair_mot_temp"
        )
    )

    learner.fit(
        dataset,
        train_split_paths=train_split_paths,
        val_split_paths=train_split_paths,
        verbose=True,
        # silent=True,
    )

    print(learner)
    print()


def test_optimize():

    dataset_path = "./perception/object_tracking_2d/datasets/data"
    train_split_paths = {
        "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train"
    }

    dataset = MotDataset(dataset_path)

    learner = ObjectTracking2DFairMotLearner(
        iters=10,
        num_epochs=2,
        checkpoint_after_iter=3,
        checkpoint_load_iter=9,
        temp_path=os.path.join(
            "tests", "sources", "tools",
            "perception", "object_tracking_2d",
            "fair_mot",
            "fair_mot_temp"
        )
    )

    learner.optimize()

    print(learner)
    print()


def test_save():

    dataset_path = "./perception/object_tracking_2d/datasets/data"
    train_split_paths = {
        "mot20": "./perception/object_tracking_2d/datasets/splits/mot20.train"
    }

    dataset = MotDataset(dataset_path)

    learner = ObjectTracking2DFairMotLearner(
        iters=10,
        num_epochs=2,
        checkpoint_after_iter=3,
        checkpoint_load_iter=9,
        temp_path=os.path.join(
            "tests", "sources", "tools",
            "perception", "object_tracking_2d",
            "fair_mot",
            "fair_mot_temp"
        )
    )
    learner.save("tests")
    learner.load("tests", verbose=True)

    learner.optimize()

    print(learner)
    print()


# test_dataset()
test_fit()
# test_optimize()
# test_save()
