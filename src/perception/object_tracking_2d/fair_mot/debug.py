
from engine.datasets import ExternalDataset
from perception.object_tracking_2d.datasets.mot_dataset import JointDataset
from perception.object_tracking_2d.fair_mot.object_tracking_2d_fair_mot_learner import ObjectTracking2DFairMotLearner

def test_dataset():

    dataset_path = "./perception/object_tracking_2d/datasets/data"
    train_split_paths = {
        "mot20": "./perception/object_tracking_2d/datasets/data/mot20.train"
    }

    dataset = JointDataset(dataset_path, train_split_paths)

    values = [dataset[0], dataset[1]]

    print(values)


def test_fit():

    dataset_path = "./perception/object_tracking_2d/datasets/data"
    train_split_paths = {
        "mot20": "./perception/object_tracking_2d/datasets/data/mot20.train"
    }

    dataset = ExternalDataset(dataset_path, "mot")

    learner = ObjectTracking2DFairMotLearner()

    learner.fit(
        dataset,
        train_split_paths=train_split_paths,
        val_split_paths=train_split_paths,
    )

    print(learner)


test_fit()
