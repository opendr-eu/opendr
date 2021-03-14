from perception.pose_estimation.lightweight_open_pose.algorithm.scripts import make_val_subset
from perception.pose_estimation.lightweight_open_pose.algorithm.datasets.coco import CocoValDataset

from perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from engine.datasets import ExternalDataset, DatasetIterator

import os


class CustomValDataset(DatasetIterator):
    """
    An example of a custom dataset class implementation.
    """
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __init__(self, path, use_subset=False, subset_name="val_subset.json", subset_size=250,
                 images_folder_default_name="val2017",
                 annotations_filename="person_keypoints_val2017.json"):
        super().__init__()
        self.path = path
        # Get files and subdirectories of dataset.path directory
        f = []
        dirs = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            f = filenames
            dirs = dirnames
            break

        # Get images folder
        if images_folder_default_name not in dirs:
            raise UserWarning("Didn't find \"" + images_folder_default_name +
                              "\" folder in the dataset path provided.")
        images_folder = os.path.join(path, images_folder_default_name)

        # Get annotations file
        if annotations_filename not in f:
            raise UserWarning("Didn't find \"" + annotations_filename +
                              "\" file in the dataset path provided.")
        val_labels_file = os.path.join(path, annotations_filename)

        if use_subset:
            val_sub_labels_file = os.path.join(path, subset_name)
            if subset_name not in f:
                print("Didn't find " + subset_name + " in dataset.path, creating new...")
                make_val_subset.make_val_subset(val_labels_file,
                                                output_path=val_sub_labels_file,
                                                num_images=subset_size)
                print("Created new validation subset file.")
                self.data = CocoValDataset(val_sub_labels_file, images_folder)
            else:
                print("Val subset already exists.")
                self.data = CocoValDataset(val_sub_labels_file, images_folder)
                if len(self.data) != subset_size:
                    print("Val subset is wrong size, creating new.")
                    # os.remove(val_sub_labels_file)
                    make_val_subset.make_val_subset(val_labels_file,
                                                    output_path=val_sub_labels_file,
                                                    num_images=subset_size)
                    print("Created new validation subset file.")
                    self.data = CocoValDataset(val_sub_labels_file, images_folder)
        else:
            self.data = CocoValDataset(val_labels_file, images_folder)


use_external_dataset = True  # Whether to use an ExternalDataset or the CustomDataset created above
parent_dir = "." + os.sep

# path_to_data: path where the data are stored. Inside this folder the val2017 folder is expected if validating on
# COCO2017, as well as the person_keypoints_val2017.json file
path_to_data = ".." + os.sep + "data"

device = "cuda"
num_refinement_stages = 2  # 0, 1, 2 or 3 refinement stages, more means better metrics and slower speed

use_validation_subset = True  # Whether to create a subset of the validation set
validation_subset_size = 250  # The number of images contained in the subset

pose_estimator = LightweightOpenPoseLearner(temp_path=parent_dir, device=device,
                                            num_refinement_stages=num_refinement_stages)
if use_external_dataset:
    validation_dataset = ExternalDataset(path=path_to_data, dataset_type="COCO")
else:
    validation_dataset = CustomValDataset(path=path_to_data, use_subset=use_validation_subset,
                                          subset_size=validation_subset_size)

pose_estimator.download(path="trainedModel", verbose=True)
pose_estimator.load("trainedModel")

pose_estimator.eval(validation_dataset,
                    use_subset=use_validation_subset, subset_size=validation_subset_size)
