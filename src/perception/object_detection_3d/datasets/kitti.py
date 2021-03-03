from engine.datasets import DatasetIterator, ExternalDataset
from perception.object_detection_3d.datasets.create_data_kitti import (
    create_kitti_info_file,
    create_reduced_point_cloud,
    create_groundtruth_database,
)
import os


class DatasetSamplerOptions:
    def __init__(self):

        super().__init__()


class KittiDataset(ExternalDataset):
    def __init__(self, path, kitti_subsets_path="./perception/object_detection_3d/datasets/kitti_subsets"):

        super().__init__(path, "kitti")

        self.path = path
        self.kitti_subsets_path = kitti_subsets_path

        self.__prepare_data()

    def __prepare_data(self):

        files = os.listdir(self.path)

        if ("gt_database" in files) and ("kitti_infos_train.pkl" in files):
            print(":::Data Ready:::")
            return

        print(":::Create KITTI Info File:::")
        create_kitti_info_file(self.path, self.kitti_subsets_path)

        print(":::Create Reduced Point Cloud:::")
        create_reduced_point_cloud(self.path)

        print(":::Create Ground-Truth Database:::")
        create_groundtruth_database(self.path)

        print(":::Data Ready:::")

        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return super().__len__()
