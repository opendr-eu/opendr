
## gem module

The *gem* module contains the *GEMLearner* class, which inherits from the abstract class *Learner*.

### Class GEMLearner
Bases: `engine.learners.Learner`

The *GEMLearner* class is a multimodal wrapper inspired by DETR [[1]](#detr-paper) object detection algorithm based on the original [DETR implementation](https://github.com/facebookresearch/detr). The fusion methodologies employed in *GEMLearner* are explained in GEM article [[2]](#gem-paper).
It can be used to perform object detection on images (inference) and train GEM object detection models.

The [GEMLearner](#src.perception.object_detection_2d.detr.detr_learner.py) class has the
following public methods:

#### `GEMLearner` constructor
```python
GEMLearner(model_config_path, dataset_config_path, iters, lr, batch_size, optimizer, backbone, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, threshold, num_classes, return_segmentations)
```

Constructor parameters:
- **model_config_path**: *str, default="OpenDR/src/perception/object_detection_2d/gem/algorithm/config/model_config.yaml"*
  Specifies the path to the config file that contains the additional parameters from the original [DETR implementation](https://github.com/facebookresearch/detr).
  -**dataset_config_path**: *str, default="OpenDR/src/perception/object_detection_2d/gem/algorithm/config/dataset_config.yaml"*
  The dataset folder structure e.g., image folders and annotation files location are defined here.
- **iters**: *int, default=10*
  Specifies the number of epochs the training should run for.
- **lr**: *float, default=1e-4*
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=1*
  Specifies number of images to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **optimizer**: *{'sdg', 'adam', 'adamw'}, default='adamw'*
  Specifies the type of optimizer that is used during training.
- **backbone**: *{'resnet50'}, default='resnet50'*
  Specifies the backbone architecture. Currently only supports *'resnet50'*.
- **checkpoint_after_iter**: *int, default=0*
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default='temp'*
  Specifies a path where the algorithm looks for pretrained backbone weights, the checkpoints are saved along with the logging files.
- **device**: *{'cpu', 'cuda'}, default='cuda'*
  Specifies the device to be used.
- **threshold**: *float, default=0.7*
  Specifies the threshold for object detection inference. An object is detected if the confidence of the output is higher than the specified threshold.
- **num_classes**: *int, default=91*
  Specifies the number of classes of the model. The default is 91, since this is the number of classes in the COCO dataset, but modifying the *num_classes* allows the user to train on its own dataset.
  It is also possible to use pretrained DETR models with the specified num_classes, since the head of the pretrained model with be modified appropriately. An example below demonstrates this.
  In this way, a model that was pretrained on the coco dataset can be finetuned to another dataset. Training on other datasets than COCO can be done by defining dataset folder structure details in *dataset_config.yaml* and using corresponding *num_classes*.
- **return_segmentations**: *bool, default=False*
  Specifies whether the model returns, next to bounding boxes, segmentations of objects. Currently this feature is not supported in GEM.

#### `GEMLearner.fit`
```python
GEMLearner.fit(m1_train_edataset, m2_train_edataset, annotations_folder, m1_train_annotations_file, m2_train_annotations_file, m1_train_images_folder, m2_train_images_folder, out_dir, trial_dir, logging_path, silent, verbose, m1_val_edataset, m2_val_edataset, m1_val_annotations_file, m2_val_annotations_file, m1_val_images_folder, m2_val_images_folder)
```
This method is used for training the algorithm on a train dataset and validating on a val dataset.
Returns a dictionary containing stats regarding the last evaluation ran.
Parameters:
- **m1_train_edataset**: *object, default=None*
    Object that holds the training dataset. Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`. If *None*, `ExternalDataset` type is assigned automatically.
- **m2_train_edataset**: *object, default=None*
	Same as *m1_train_edataset*.
- **annotations_folder** : *str, default=None*
    Foldername of the annotations json file. This folder may only be provided in the *dataset_config.yaml* file.
- **m1_train_annotations_file** : *str, default=None*
    Filename of the train annotations json file. This file may only be provided in the *dataset_config.yaml* file.
- **m2_train_annotations_file** : *str, default=None*
    Same as *m1_train_annotations_file*.
- **m1_train_images_folder** : *str, default=None*
    Name of the folder that contains the train dataset images. This folder may only be provided in the *dataset_config_yaml* file. Note that this is a folder (or a nested-folder) name, not a path.
- **m2_train_images_folder** : *str, default=None*
    Same as *m1_train_images_folder*.
- **out_dir**: *str, default='outputs'*
	Output root folder to save the outputs of training sessions initiated by *fit* function.
- **trial_dir**: *str, default='trial'*
	Each training session checkpoints are saved in here. If the folder already exists, a new folder is created with current date and time appended to the name of the existing one for the new training trial.
- **logging_path** : *str, default=''*
    Path to save tensorboard log files. If set to None or '', tensorboard logging is disabled.
- **silent** : *bool, default=False*
    If True, all printing of training progress reports and other information to STDOUT are disabled.
- **verbose** : *bool, default=True*
    Enables the maximum verbosity.
- **m1_val_edataset**: *object, default=None*
    Object that holds the training dataset. Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`. If *None*, `ExternalDataset` type is assigned automatically.
- **m2_val_edataset**: *object, default=None*
	Same as *m1_val_edataset*.
- **m1_val_annotations_file** : *str, default=None*
    Filename of the train annotations json file. This file may only be provided in the *dataset_config.yaml* file.
- **m2_val_annotations_file** : *str, default=None*
    Same as *m1_val_annotations_file*.
- **m1_val_images_folder** : *str, default=None*
    Name of the folder that contains the train dataset images. This folder may only be provided in the *dataset_config.yaml* file. Note that this is a folder (or a nested-folder) name, not a path.
- **m2_val_images_folder** : *str, default=None*
    Same as *m1_val_images_folder*.

#### `GEMLearner.eval`
```python
GEMLearner.eval(self, m1_edataset, m2_edataset, m1_images_folder, m2_images_folder, annotations_folder, m1_annotations_file, m2_annotations_file)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.
Parameters:
- **m1_edataset** : *object, default=None*
    ExternalDataset class object or DatasetIterator class object. Object that holds the evaluation dataset. If *None*, `ExternalDataset` type is assigned automatically.
- **m2_edataset** : *object*
    Same as *m1_edataset*.
- **m1_images_folder** : *str, default='val2017'*
    Folder name that contains the dataset images. This folder may only be provided in the *dataset_config.yaml* file. Note that this is a folder (or a nested-folder) name, not a path.
- **m2_images_folder** : *str, default='val2017'*
    Same as *m1_images_folder*.
- **annotations_folder** : *str, default='Annotations'*
    Folder name of the annotations json file. This folder may only be provided in the *dataset_config.yaml* file.
- **m1_annotations_file** : *str, default='instances_val2017.json'*
    Filename of the annotations json file. This file may only be provided in the *dataset_config.yaml* file.
- **m2_annotations_file** : *str, default='instances_val2017.json'*
    Same as *m1_annotations_file*.

#### `GEMLearner.infer`
```python
GEMLearner.infer(self, m1_image, m2_image)
```

This method is used to perform object detection on an image.
Returns an engine.target.BoundingBoxList object, which contains bounding boxes that are described by the left-top corner and its width and height, or returns an empty list if no detections were made.

Parameters:
- **m1_image : *object*
    Image of type engine.data.Image class or np.array. Image to run inference on.
- **m2_image : *object*
    Same as *m1_image*.

#### `GEMLearner.save`
```python
GEMLearner.save(self, path)
```
This method is used to save a trained model.
Provided with the path, it creates the "name" directory, if it does not already
exist. Inside this folder, the model is saved as *"name.pth"* and the metadata file as *"name".json*. If the directory already exists, the *"name.pth"* and *"name".json* files are overwritten.

Parameters:
- **path**: *str*
  Path to save the model, including the filename.

#### `GEMLearner.load`
```python
GEMLearner.load(self, path)
```
This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:
- **path**: *str*
  Path of the model to be loaded.

  #### `GEMLearner.download`
  ```python
  GEMLearner.download(self, path, mode, verbose)
  ```
  Download utility for downloading pretrained models and test data.

  Parameters:
  - **path** : *str*
    Determines the path to the location where the downloaded files will be stored.
  - **mode** : *str*
    Determines the files that will be downloaded. Valid values are: "weights_detr", "pretrained_detr", "pretrained_gem", "test_data_l515" and "test_data_sample_images".
    In case of "weights_detr", the weigths for single modal DETR with *resnet50* backbone are downloaded. In case of "pretrained_detr", the weigths for single modal pretrained DETR with *resnet50* backbone are downloaded. In case of "pretrained_gem", the weights from *'gem_scavg_e294_mAP0983_rn50_l515_7cls.pth'* (backbone: *'resnet50'*, fusion_method: *'scalar averaged'*, trained on *RGB-Infrared l515_dataset* are downloaded.
    In case of "test_data_l515", the *RGB-Infrared l515* dataset is downloaded from the OpenDR server.
    In case of "test_data_sample images", two sample images for testing the *infer* function are downloaded.
  - **verbose** : *bool*
    Enables the maximum verbosity.

#### Examples

* **Training example:.**
  The details of multimodal training and evaluation dataset should be described in the *dataset_config.yaml* file.   The `batch_size` argument should be adjusted according to available memory.

```python
from opendr.perception.object_detection_2d.gem.gem_learner import GEMLearner

learner = GEMLearner(iters=1, batch_size=1, num_classes=7)
learner.fit()
```

To load pretrained weights from single modal DETR trained on COCO dataset and save after the training is finished:
```python
from opendr.perception.object_detection_2d.gem.gem_learner import GEMLearner

learner = GEMLearner(iters=1, batch_size=1, num_classes=7)
learner.create_model(pretrained='detr_coco')
learner.fit()
learner.save('./saved_models/trained_model')
```

* **Inference and result drawing example on the sample images can be downloaded from OpenDR server, similar to single modal inference in [detr_demo colab].(https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=Jf59UNQ37QhJ).**
	This example shows how to perform inference on sample images and draw the resulting bounding boxes using a gem model that is pretrained on the *RGB-Infrared l515 datset*.
```python
from opendr.perception.object_detection_2d.gem.gem_learner import GEMLearner
from PIL import Image

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")

# l515_dataset classes
classes = ['chair', 'cycle', 'bin', 'laptop', 'drill', 'rocker']

# colors for visualization
colors = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Function for plotting results
def plot_results(pil_img, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for box, c in zip(boxes, colors*100):
        ax.add_patch(plt.Rectangle((box.left, box.top), box.width, box.height,
                                   fill=False, color=c, linewidth=3))
        text = f'{classes[box.name-1]}: {box.confidence:0.2f}'
        ax.text(box.left, box.top, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

learner = GEMLearner(num_classes=7)
learner.fusion_method = 'sc_avg'
learner.create_model(pretrained='gem_l515')

learner.download_sample_images(path='./sample_images')
m1_img = Image.open('./sample_images/rgb/2021_04_22_21_35_47_852516.jpg')
m2_img = Image.open('./sample_images/aligned_infra/2021_04_22_21_35_47_852516.jpg')

bounding_box_list = learner.infer(m1_img, m2_img)
plot_results(m1_img, bounding_box_list)
```
#### References
<a name="detr-paper" href="https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers">[1]</a> End-to-end Object Detection with Transformers,
[arXiv](https://arxiv.org/abs/2005.12872).

<a name="gem-paper">[2]</a> GEM: Glare or Gloom, I Can Still See You -- End-to-End Multimodal Object Detection
[arXiv](https://arxiv.org/abs/2102.12319).
