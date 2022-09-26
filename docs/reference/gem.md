
## gem module

The *gem* module contains the *GemLearner* class, which inherits from the abstract class *Learner*.

### Class GemLearner
Bases: `engine.learners.Learner`

The *GemLearner* class is a multimodal wrapper inspired by DETR [[1]](#detr-paper) object detection algorithm based on the original [DETR implementation](https://github.com/facebookresearch/detr).
The fusion methodologies employed in *GemLearner* are explained in GEM article [[2]](#gem-paper).
It can be used to perform object detection on images (inference) and train GEM object detection models.

The [GemLearner](/src/opendr/perception/object_detection_2d/detr/detr_learner.py) class has the following public methods:

#### `GemLearner` constructor
```python
GemLearner(self, model_config_path, dataset_config_path, iters, lr, batch_size, optimizer, backbone, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, threshold, num_classes, panoptic_segmentation)
```

Constructor parameters:
- **model_config_path**: *str, default="OpenDR/src/perception/object_detection_2d/gem/algorithm/config/model_config.yaml"*
  Specifies the path to the config file that contains the additional parameters from the original [DETR implementation](https://github.com/facebookresearch/detr).
- **dataset_config_path**: *str, default="OpenDR/src/perception/object_detection_2d/gem/algorithm/config/dataset_config.yaml"*
  The dataset folder structure e.g., image folders and annotation files location are defined here.
- **iters**: *int, default=10*
  Specifies the number of epochs the training should run for.
- **lr**: *float, default=1e-4*
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=1*
  Specifies number of images to be bundled up in a batch during training.
  This heavily affects memory usage, adjust according to your system.
- **optimizer**: *{'sgd', 'adam', 'adamw'}, default='adamw'*
  Specifies the type of optimizer that is used during training.
- **backbone**: *{'resnet50'}, default='resnet50'*
  Specifies the backbone architecture.
  Currently only supports *'resnet50'*.
- **checkpoint_after_iter**: *int, default=0*
  Specifies per how many training iterations a checkpoint should be saved.
  If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*
  Specifies which checkpoint should be loaded.
  If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default='temp'*
  Specifies a path where the algorithm looks for pretrained backbone weights, the checkpoints are saved along with the logging files.
- **device**: *{'cpu', 'cuda'}, default='cuda'*
  Specifies the device to be used.
- **threshold**: *float, default=0.7*
  Specifies the threshold for object detection inference.
  An object is detected if the confidence of the output is higher than the specified threshold.
- **num_classes**: *int, default=91*
  Specifies the number of classes of the model.
  The default is 91, since this is the number of classes in the COCO dataset, but modifying the *num_classes* allows the user to train on its own dataset.
  It is also possible to use pretrained DETR models with the specified *num_classes*, since the head of the pretrained model with be modified appropriately.
  An example below demonstrates this.
  In this way, a model that was pretrained on the coco dataset can be finetuned to another dataset.
  Training on other datasets than COCO can be done by defining dataset folder structure details in *dataset_config.yaml* and using corresponding *num_classes*.
- **panoptic_segmentation**: *bool, default=False*
  Specifies whether the model returns, next to bounding boxes, segmentations of objects.
  Currently this feature is not supported in GEM.

#### `GemLearner.fit`
```python
GemLearner.fit(self, m1_train_edataset, m2_train_edataset, annotations_folder, m1_train_annotations_file, m2_train_annotations_file, m1_train_images_folder, m2_train_images_folder, out_dir, trial_dir, logging_path, silent, verbose, m1_val_edataset, m2_val_edataset, m1_val_annotations_file, m2_val_annotations_file, m1_val_images_folder, m2_val_images_folder)
```
This method is used for training the algorithm on a train dataset and validating on a val dataset.
Returns a dictionary containing stats regarding the last evaluation ran.

Parameters:
- **m1_train_edataset**: *object, default=None*
    Object that holds the training dataset.
    Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
    If *None*, `ExternalDataset` type is assigned automatically.
- **m2_train_edataset**: *object, default=None*
	Same as *m1_train_edataset*.
- **annotations_folder** : *str, default=None*
    Foldername of the annotations json file.
    This folder may only be provided in the *dataset_config.yaml* file.
- **m1_train_annotations_file** : *str, default=None*
    Filename of the train annotations json file.
    This file may only be provided in the *dataset_config.yaml* file.
- **m2_train_annotations_file** : *str, default=None*
    Same as *m1_train_annotations_file*.
- **m1_train_images_folder** : *str, default=None*
    Name of the folder that contains the train dataset images.
    This folder may only be provided in the *dataset_config.yaml* file.
    Note that this is a folder (or a nested-folder) name, not a path.
- **m2_train_images_folder** : *str, default=None*
    Same as *m1_train_images_folder*.
- **out_dir**: *str, default='outputs'*
	Output root folder to save the outputs of training sessions initiated by *fit* function.
- **trial_dir**: *str, default='trial'*
	Each training session checkpoints are saved in here.
  If the folder already exists, a new folder is created with current date and time appended to the name of the existing one for the new training trial.
- **logging_path** : *str, default=''*
    Path to save Tensorboard log files.
    If set to None or '', Tensorboard logging is disabled.
- **silent** : *bool, default=False*
    If True, all printing of training progress reports and other information to STDOUT are disabled.
- **verbose** : *bool, default=True*
    Enables the maximum verbosity.
- **m1_val_edataset**: *object, default=None*
    Object that holds the training dataset.
    Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
    If *None*, `ExternalDataset` type is assigned automatically.
- **m2_val_edataset**: *object, default=None*
	Same as *m1_val_edataset*.
- **m1_val_annotations_file** : *str, default=None*
    Filename of the train annotations json file.
    This file may only be provided in the *dataset_config.yaml* file.
- **m2_val_annotations_file** : *str, default=None*
    Same as *m1_val_annotations_file*.
- **m1_val_images_folder** : *str, default=None*
    Name of the folder that contains the train dataset images.
    This folder may only be provided in the *dataset_config.yaml* file.
    Note that this is a folder (or a nested-folder) name, not a path.
- **m2_val_images_folder** : *str, default=None*
    Same as *m1_val_images_folder*.

#### `GemLearner.eval`
```python
GemLearner.eval(self, m1_edataset, m2_edataset, m1_images_folder, m2_images_folder, annotations_folder, m1_annotations_file, m2_annotations_file, verbose)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:
- **m1_edataset** : *object, default=None*
    ExternalDataset class object or DatasetIterator class object.
    Object that holds the evaluation dataset.
    If *None*, `ExternalDataset` type is assigned automatically.
- **m2_edataset** : *object, default=None*
    Same as *m1_edataset*.
- **m1_images_folder** : *str, default='m1_val2017'*
    Folder name that contains the dataset images.
    This folder may only be provided in the *dataset_config.yaml* file.
    Note that this is a folder (or a nested-folder) name, not a path.
- **m2_images_folder** : *str, default='m2_val2017'*
    Same as *m1_images_folder*.
- **annotations_folder** : *str, default='Annotations'*
    Folder name of the annotations json file.
    This folder may only be provided in the *dataset_config.yaml* file.
- **m1_annotations_file** : *str, default='m1_instances_val2017.json'*
    Filename of the annotations json file.
    This file may only be provided in the *dataset_config.yaml* file.
- **m2_annotations_file** : *str, default='m2_instances_val2017.json'*
    Same as *m1_annotations_file*.
- **verbose** : *bool, default=True*
    Enables the maximum verbosity.

#### `GemLearner.infer`
```python
GemLearner.infer(self, m1_image, m2_image)
```

This method is used to perform object detection on an image.
Returns an engine.target.BoundingBoxList object, which contains bounding boxes that are described by the left-top corner
and its width and height, or returns an empty list if no detections were made.
Also returns the weights of the two modalities.

Parameters:
- **m1_image** : *object*
    Image of type engine.data.Image class or np.array.
    Image to run inference on.
- **m2_image** : *object*
    Same as *m1_image*.

#### `GemLearner.save`
```python
GemLearner.save(self, path, verbose)
```
This method is used to save a trained model.
Provided with the path, it creates the "name" directory, if it does not already exist.
Inside this folder, the model is saved as *"name.pth"* and the metadata file as *"name".json*.
If the directory already exists, the *"name.pth"* and *"name".json* files are overwritten.

Parameters:
- **path**: *str*
  Path to save the model, including the filename.
- **verbose**: *bool, default=False*
  Enables the maximum verbosity.


#### `GemLearner.load`
```python
GemLearner.load(self, path, verbose)
```
This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata.json file included.

Parameters:
- **path**: *str*
  Path of the model to be loaded.
- **verbose**: *bool, default=False*
  Enables maximum verbosity

#### `GemLearner.download`
```python
GemLearner.download(self, path, mode, verbose)
```
Download utility for downloading pretrained models and test data.

Parameters:
- **path** : *str, default=None*
  Determines the path to the location where the downloaded files will be stored.
- **mode** : *str, default='pretrained_gem'*
  Determines the files that will be downloaded.
  Valid values are: "weights_detr", "pretrained_detr", "pretrained_gem", "test_data_l515" and "test_data_sample_images".
  In case of "weights_detr", the weigths for single modal DETR with *resnet50* backbone are downloaded.
  In case of "pretrained_detr", the weigths for single modal pretrained DETR with *resnet50* backbone are downloaded.
  In case of "pretrained_gem", the weights (backbone: *'resnet50' or 'mobilenetv2'*, fusion_method: *'scalar averaged'*, trained on *RGB-Infrared l515_dataset*) are downloaded.
  In case of "test_data_l515", the *RGB-Infrared l515* dataset is downloaded from the OpenDR server.
  In case of "test_data_sample images", two sample images for testing the *infer* function are downloaded.
- **verbose** : *bool, default=False*
  Enables the maximum verbosity.

#### Demo and Tutorial

An inference [demo](../../projects/python/perception/object_detection_2d/gem/inference_demo.py) and
[tutorial](../../projects/python/perception/object_detection_2d/gem/inference_tutorial.ipynb) are available.

#### Examples

* **Training example:**
  The details of multimodal training and evaluation dataset should be described in the *dataset_config.yaml* file.
  The `batch_size` argument should be adjusted according to available memory.

```python
from opendr.perception.object_detection_2d import GemLearner

learner = GemLearner(iters=1, batch_size=1, num_classes=7)
learner.fit()
```

To load pretrained weights from single modal DETR trained on COCO dataset and save after the training is finished:
```python
from opendr.perception.object_detection_2d.gem.gem_learner import GemLearner

learner = GemLearner(iters=1, batch_size=1, num_classes=7)
learner.download(mode='pretrained_detr')
learner.fit()
learner.save('./saved_models/trained_model')
```

* **Inference and result drawing example on the sample images can be downloaded from OpenDR server, similar to single modal inference in [detr_demo colab](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=Jf59UNQ37QhJ).**
	This example shows how to perform inference on sample images and draw the resulting bounding boxes using a gem model that is pretrained on the *RGB-Infrared l515 datset*.

```python
from opendr.perception.object_detection_2d import GemLearner
from opendr.perception.object_detection_2d.gem.algorithm.util.draw import draw
from opendr.engine.data import Image
import cv2

# First we initialize the learner
learner = GemLearner(num_classes=7, device='cuda')
# Next, we download a pretrained model
learner.download(mode='pretrained_gem')
# And some sample images
learner.download(mode='test_data_sample_images')
# We now read the sample images
m1_img = Image.open('temp/sample_images/rgb/2021_04_22_21_35_47_852516.jpg')
m2_img = Image.open('temp/sample_images/aligned_infra/2021_04_22_21_35_47_852516.jpg')
# Perform inference
bounding_box_list, w_sensor1, _ = learner.infer(m1_img, m2_img)
# Visualize the detections
# The blue/green bar shows the weights of the two modalities
# Fully blue means relying purely on the first modality
# Fully green means relying purely on the second modality
cv2.imshow('Detections', draw(m1_img.opencv(), bounding_box_list, w_sensor1))
cv2.waitKey(0)
```

#### Performance Evaluation

The performance evaluation results of the *GemLearner* are reported in the Table below.
These tests have been performed on several platforms, i.e. a laptop CPU, laptop GPU and on the Jetson TX2.
Also, the results for two different backbones are shown, i.e. ResNet-50 and MobileNet-v2.
Inference was performed on a 1280x720 RGB image and a 1280x720 infrared image.

| Method       | CPU i7-10870H  (FPS) | RTX 3080 Laptop (FPS) | Jetson TX2 (FPS) |
|--------------|:--------------------:|:---------------------:|:----------------:|
| ResNet-50    |         0.87         |         11.83         |       0.83       |
| MobileNet-v2 |         2.44         |         18.70         |       2.19       |


Apart from the inference speed, which is reported in FPS, we also report the memory usage, as well as energy consumption on a reference platform in the Table below.
Again, inference was performed on a 1280x720 RGB image and a 1280x720 infrared image.

| Method       | Memory (MB) RTX 3080 Laptop | Energy (Joules)  - Total per inference Jetson TX2 |
|--------------|:---------------------------:|:-------------------------------------------------:|
| ResNet-50    |            1853.4           |                        28.2                       |
| MobileNet-v2 |            931.2            |                        8.4                        |

Below, the performance of GEM is in terms of accuracy is presented.
These results were obtained on the L515-Indoor dataset, which can also be downloaded using the GemLearner class from OpenDR.

| Method       | Mean Average Precision | Energy (Joules)  - Total per inference Jetson TX2 |
|--------------|:----------------------:|:-------------------------------------------------:|
| Resnet-50    |          0.982         |                        28.2                       |
| MobileNet-v2 |          0.833         |                        8.4                        |

The platform compatibility evaluation is also reported below:

|                   Platform                   | Test results |
|----------------------------------------------|:------------:|
| x86 - Ubuntu 20.04 (bare installation - CPU) |     Pass     |
| x86 - Ubuntu 20.04 (bare installation - GPU) |     Pass     |
| x86 - Ubuntu 20.04 (pip installation)        |     Pass     |
| x86 - Ubuntu 20.04 (CPU docker)              |     Pass     |
| x86 - Ubuntu 20.04 (GPU docker)              |     Pass     |
| NVIDIA Jetson TX2                            |     Pass     |

#### References
<a name="detr-paper" href="https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers">[1]</a> Carion N., Massa F., Synnaeve G., Usunier N., Kirillov A., Zagoruyko S. (2020) End-to-End Object Detection with Transformers. In: Vedaldi A., Bischof H., Brox T., Frahm JM. (eds) Computer Vision – ECCV 2020. ECCV 2020. Lecture Notes in Computer Science, vol 12346. Springer, Cham. [doi](https://doi.org/10.1007/978-3-030-58452-8_13),
[arXiv](https://arxiv.org/abs/2005.12872).

<a name="gem-paper">[2]</a> Mazhar, O., Babuska, R., & Kober, J. (2021). GEM: Glare or Gloom, I Can Still See You - End-to-End Multi-Modal Object Detection. IEEE Robotics and Automation Letters, 6(4), 6321-6328. [doi](https://doi.org/10.1109/LRA.2021.3093871), [arXiv](https://arxiv.org/abs/2102.12319).
