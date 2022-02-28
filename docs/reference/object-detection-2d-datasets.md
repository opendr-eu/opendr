# Object Detection 2D Datasets

## Base Classes

### DetectionDataset base class

Bases: `engine.datasets.DatasetIterator`

The *DetectionDataset* class inherits from the *DatasetIterator* class and extends it with functions and properties aimed at 2d Object Detection datasets. Each *DetectionDataset* object must be initialized with the following parameters:

- **classes**: *list*\
  List of class names of the training dataset.
- **dataset_type**: *str*\
  Dataset type, i`.e., an assigned name.
- **root**: *str*\
  Path to dataset root directory.
- **image_paths**: *list, default=None*\
  List of image data paths.
- **splits**: *str, default=''*\
  List of dataset splits to load (e.g., train/val).
- **image_transform**: *callable object, default=None*\
  Transformation to apply to images. Intended for image format transformations.
- **target_transform**: *callable object, default=None*\
  Transformation to apply to bounding boxes. Intended for formatting the bounding boxes for each detector.
- **transform**: *callable object, default=None*\
  Transformation to apply to both images and bounding boxes. Intended for data augmentation purposes.
  
Methods:

#### `DetectionDataset.set_transform`
Setter for the internal **transform** object/function.

#### `DetectionDataset.set_image_transform`
Setter for the internal **image_transform** object/function.

#### `DetectionDataset.set_target_transform`
Setter for the internal **target_transform** object/function.

#### `DetectionDataset.transform`
Returns the `DetectionDataset` wrapped as a `MappedDetectionDataset`, where the data is transformed according to the argument callable object/function. This function ensures fit/eval compatibility between `DetectionDataset` and `ExternalDataset` for [GluonCV](https://github.com/dmlc/gluon-cv) based detectors.

#### `DetectionDataset.get_image`
Returns an image from the dataset. Intended for test sets without annotations.

#### `DetectionDataset.get_bboxes`
Returns the bounding boxes for a given sample.


### MappedDetectionDataset class

Bases: `engine.datasets.DatasetIterator`

This class wraps any `DetectionDataset` and applies `map_function` to the data.

### ConcatDataset class

Bases: `perception.object_detection_2d.datasets.DetetionDataset`

Returns a new `DetectionDataset` which is a concatenation of the `datasets` param. The datasets are assumed to have the same classes.

### XMLBasedDataset class

Bases: `perception.object_detection_2d.datasets.DetetionDataset`

This class is intended for any dataset in PASCAL VOC .xml format, making it compatible with datasets annotated using the [labelImg](https://github.com/tzutalin/labelImg) tool. Each *XMLBasedDataset* object must be initialized with the following parameters:

- **dataset_type**: *str*\
  Dataset type, i.e., assigned name.
- **root**: *str*\
  Path to dataset root directory.
- **classes**: *list, default=None*\
  Class names. If None, they will be inferred from the annotations.
- **splits**: *str, default=''*\
  List of dataset splits to load (e.g., train/val).
- **image_transform**: *callable object, default=None*\
  Transformation to apply to images. Intended for image format transformations.
- **target_transform**: *callable object, default=None*\
  Transformation to apply to bounding boxes. Intended for formatting the bounding boxes for each detector.
- **transform**: *callable object, default=None*\
  Transformation to apply to both images and bounding boxes. Intended for data augmentation purposes.
- **images_dir**: *str, default='images'*\
  Name of subdirectory containing dataset images.
- **annotations_dir**: *str, default='annotations'*\
  Name of subdirectory containing dataset annotations.
- **preload_anno**: *bool, default=False*\
  Whether to preload annotations, for datasets that fit in memory.
  
