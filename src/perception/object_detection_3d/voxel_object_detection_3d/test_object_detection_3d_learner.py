from engine.datasets import ExternalDataset
from perception.object_detection_3d.datasets.kitti import KittiDatasetIterator
from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import VoxelObjectDetection3DLearner
from perception.object_detection_3d.voxel_object_detection_3d.tanet_configs import TANET_16

dataset_path = "/data/sets/opendr_kitti"
tanet_path = "./perception/object_detection_3d/models/tanet_16_car"
dataset = ExternalDataset(dataset_path, "kitti")

iterator = KittiDatasetIterator(dataset)

learner = VoxelObjectDetection3DLearner(model_config=TANET_16)
learner.load(tanet_path)



print(learner)