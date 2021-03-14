from torchvision import transforms

from perception.pose_estimation.lightweight_open_pose.algorithm.scripts import \
    prepare_train_labels, make_val_subset
from perception.pose_estimation.lightweight_open_pose.algorithm.datasets.coco import \
    CocoTrainDataset, CocoValDataset
from perception.pose_estimation.lightweight_open_pose.algorithm.datasets.transformations import \
    ConvertKeypoints, Scale, Rotate, CropPad, Flip

from perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from engine.datasets import ExternalDataset, DatasetIterator

import os


class CustomTrainDataset(DatasetIterator):
    """
    An example of a custom dataset class implementation.
    """
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __init__(self, path, images_folder_default_name="train2017",
                 annotations_filename="person_keypoints_train2017.json",
                 prepared_annotations_name="prepared_train_annotations.pkl"):
        self.path = path
        super().__init__()
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
        annotations_file = os.path.join(path, annotations_filename)

        # Convert annotations to internal format if needed
        if prepared_annotations_name not in f:
            print("Didn't find " + prepared_annotations_name + " in dataset.path, creating new...")
            prepare_train_labels.convert_annotations(annotations_file,
                                                     output_path=os.path.join(path, prepared_annotations_name))
            print("Created new .pkl file containing prepared annotations in internal format.")
        prepared_train_labels = os.path.join(path, prepared_annotations_name)

        sigma = 7
        paf_thickness = 1
        stride = 8
        self.data = CocoTrainDataset(prepared_train_labels, images_folder,
                                     stride, sigma, paf_thickness,
                                     transform=transforms.Compose([
                                         ConvertKeypoints(),
                                         Scale(),
                                         Rotate(pad=(128, 128, 128)),
                                         CropPad(pad=(128, 128, 128)),
                                         Flip()]))


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


# Logging with (windows)
# tensorboard --logdir=ENTER_ABSOLUTE_PATH/parent_dir/logs/fit --host localhost --port 8088
# visit http://localhost:8088/
# Logging with (linux)
# python3 -m tensorboard.main --logdir=ENTER_ABSOLUTE_PATH/parent_dir/logs/fit --host localhost --port 8088
# visit http://localhost:8088/

if __name__ == '__main__':
    use_external_dataset = True  # Whether to use ExternalDataset or CustomDataset created above

    # parent_dir: directory where the checkpoints and logs are saved, also expecting the mobilenet pre-trained weights
    parent_dir = "." + os.sep

    # path_to_data: path where the data are stored. Inside this folder the train2017 and val2017 folders are
    # expected if training and validating on COCO2017, as well as the person_keypoints_train2017.json and
    # person_keypoints_val2017.json files
    path_to_data = ".." + os.sep + "data"
    tensorboardLogPath = ""  # Add a path to enable logging

    device = "cuda"
    batch_size = 1  # Adjust according to your RAM
    num_refinement_stages = 0  # Adjust according to training instructions
    checkpoint_load_iter = 0  # Adjust to load a certain checkpoint

    checkpoint_after_iter = 5000  # How often to save a checkpoint
    val_after = 5000  # How often to run validation
    log_after = 100  # How often to log information

    # Whether to create a subset of the validation set, use this to do quick validations while training
    use_validation_subset = True
    validation_subset_size = 250

    pose_estimator = LightweightOpenPoseLearner(temp_path=parent_dir, batch_size=batch_size, device=device,
                                                num_refinement_stages=num_refinement_stages,
                                                checkpoint_load_iter=checkpoint_load_iter,
                                                checkpoint_after_iter=checkpoint_after_iter, val_after=val_after,
                                                log_after=log_after)
    if use_external_dataset:
        training_dataset = ExternalDataset(path=path_to_data, dataset_type="COCO")
        validation_dataset = ExternalDataset(path=path_to_data, dataset_type="COCO")
    else:
        training_dataset = CustomTrainDataset(path=path_to_data)
        validation_dataset = CustomValDataset(path=path_to_data, use_subset=use_validation_subset,
                                              subset_size=validation_subset_size)

    pose_estimator.fit(training_dataset, val_dataset=validation_dataset, logging_path=tensorboardLogPath,
                       use_val_subset=use_validation_subset, val_subset_size=validation_subset_size)
