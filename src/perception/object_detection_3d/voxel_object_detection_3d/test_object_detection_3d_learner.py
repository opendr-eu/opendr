from engine.datasets import ExternalDataset
from perception.object_detection_3d.datasets.kitti import KittiDataset
from perception.object_detection_3d.voxel_object_detection_3d.voxel_object_detection_3d_learner import VoxelObjectDetection3DLearner

dataset_path = "/data/sets/opendr_kitti"
tanet_path = "./perception/object_detection_3d/voxel_object_detection_3d/models/learning_tanet_16_car"
tanet_config_path = "./perception/object_detection_3d/voxel_object_detection_3d/second/configs/tanet/car/xyres_16.proto"
dataset = KittiDataset(dataset_path)

learner = VoxelObjectDetection3DLearner(model_config_path=tanet_config_path)
learner.load(tanet_path)
learner.fit(dataset)





print(learner)