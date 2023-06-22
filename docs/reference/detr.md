## detr module

The *detr* module contains the *DetrLearner* class, which inherits from the abstract class *Learner*.

### Class DetrLearner
Bases: `engine.learners.Learner`

The *DetrLearner* class is a wrapper of the DETR [[1]](#detr-paper) object detection algorithm based on the original
[DETR implementation](https://github.com/facebookresearch/detr).
It can be used to perform object detection on images (inference) and train DETR object detection models.

The [DetrLearner](../../src/opendr/perception/object_detection_2d/detr/detr_learner.py) class has the
following public methods:

#### `DetrLearner` constructor
```python
DetrLearner(self, model_config_path, iters, lr, batch_size, optimizer, backbone, checkpoint_after_iter, checkpoint_load_iter,
temp_path, device, threshold, num_classes, panoptic_segmentation)
```

Constructor parameters:

- **model_config_path**: *str, default="OpenDR/src/perception/object_detection_2d/detr/algorithm/config/model_config.yaml"*\
  Specifies the path to the config file that contains the additional parameters from the original
  [DETR implementation](https://github.com/facebookresearch/detr).
- **iters**: *int, default=10*\
  Specifies the number of epochs the training should run for.
- **lr**: *float, default=1e-4*\
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=1*\
  Specifies number of images to be bundled up in a batch during training.
  This heavily affects memory usage, adjust according to your system.
- **optimizer**: *{'sgd', 'adam', 'adamw'}, default='adamw'*\
  Specifies the type of optimizer that is used during training.
- **backbone**: *{'resnet50', 'resnet101'}, default='resnet50'*\
  Specifies the backbone architecture.
  Other Torchvision backbones are also valid, but have no pretrained DETR models available.
  Therefore other backbone models have to be learned from scratch.
- **checkpoint_after_iter**: *int, default=0*\
  Specifies per how many training iterations a checkpoint should be saved.
  If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies which checkpoint should be loaded.
  If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default='temp'*\
  Specifies a path where the algorithm looks for pretrained backbone weights, the checkpoints are saved along with the logging
  files.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **threshold**: *float, default=0.7*\
  Specifies the threshold for object detection inference.
  An object is detected if the confidence of the output is higher than the specified threshold.
- **num_classes**: *int, default=91*\
  Specifies the number of classes of the model.
  The default is 91, since this is the number of classes in the COCO dataset, but modifying the *num_classes* allows the user to
  train on its own dataset.
  It is also possible to use pretrained DETR models with the specified *num_classes*, since the head of the pretrained model
  with be modified appropriately.
  In this way, a model that was pretrained on the coco dataset can be finetuned to another dataset.
  Training on other datasets than COCO can be done by creating a `DatasetIterator` that outputs (`Image`, `BoundingBoxList`)
  tuples.
  Below you can find an example that shows how you can create such a `DatasetIterator`.
- **panoptic_segmentations**: *bool, default=False*\
  Specifies whether panoptic segmentation is performed.
  If *True*, the `download()` method will download COCO panoptic models and the model returns, next to bounding boxes,
  segmentations of objects.

#### `DetrLearner.fit`
```python
DetrLearner.fit(self, dataset, val_dataset, logging_path, silent, verbose, annotations_folder, train_images_folder,
train_annotations_file, val_images_folder, val_annotations_file)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Returns a dictionary containing stats regarding the last evaluation ran.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **val_dataset** : *object, default=None*\
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
  Object that holds the validation dataset.
- **logging_path** : *str, default=''*\
  Path to save TensorBoard log files.
  If set to *None* or *''*, TensorBoard logging is disabled.
- **silent** : *bool, default=False*\
  If *True*, all printing of training progress reports and other information to STDOUT are disabled.
- **verbose** : *bool, default=True*\
  Enables the maximum verbosity.
- **annotations_folder** : *str, default='Annotations'*\
  Folder name of the annotations json file.
  This folder should be contained in the dataset path provided.
- **train_images_folder** : *str, default='train2017'*\
  Name of the folder that contains the train dataset images.
  This folder should be contained in the dataset path provided.
  Note that this is a folder name, not a path.
- **train_annotations_file** : *str, default='instances_train2017.json'*\
  Filename of the train annotations json file.
  This file should be contained in the dataset path provided.
-  **val_images_folder** : *str, default='val2017'*\
  Folder name that contains the validation images.
  This folder should be contained in the dataset path provided.
  Note that this is a folder name, not a path.
- **val_annotations_file** : *str, default='instances_val2017.json'*\
  Filename of the validation annotations json file.
  This file should be contained in the dataset path provided in the annotations folder provided.

#### `DetrLearner.eval`
```python
DetrLearner.eval(self, dataset, images_folder, annotations_folder, annotations_file)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:

- **dataset** : *object*\
  `ExternalDataset` class object or `DatasetIterator` class object.
  Object that holds the evaluation dataset.
- **images_folder** : *str, default='val2017'*\
  Folder name that contains the dataset images.
  This folder should be contained in the dataset path provided.
  Note that this is a folder name, not a path.
- **annotations_folder** : *str, default='Annotations'*\
  Folder name of the annotations json file.
  This file should be contained in the dataset path provided.
- **annotations_file** : *str, default='instances_val2017.json'*\
  Filename of the annotations json file.
  This file should be contained in the dataset path provided.
- **verbose** : *bool, default=True*\
  Enables the maximum verbosity.

#### `DetrLearner.infer`
```python
DetrLearner.infer(self, image)
```

This method is used to perform object detection on an image.
Returns an `engine.target.BoundingBoxList` object, which contains bounding boxes that are described by the left-top corner and
its width and height, or returns an empty list if no detections were made.

Parameters:
- **image** : *object*\
  Image of type `engine.data.Image` class or `np.array`.
  Image to run inference on.

#### `DetrLearner.save`
```python
DetrLearner.save(self, path, verbose)
```

This method is used to save a trained model.
Provided with the path, it creates the "name" directory, if it does not already exist.
Inside this folder, the model is saved as *"detr_[backbone_model].pth"* and the metadata file as *"detr_[backbone].json"*.
If the directory already exists, the *"detr_[backbone_model].pth"* and *"detr_[backbone].json"* files are overwritten.

If [`self.optimize`](/src/opendr/perception/object_detection_2d/detr/detr_learner.py#L599) was run previously, it saves the optimized ONNX model in a similar fashion with an
*".onnx"* extension, by copying it from the *self.temp_path* it was saved previously during conversion.

Parameters:

- **path**: *str*\
  Path to save the model, including the filename.
- **verbose**: *bool, default=False*\
  Enables the maximum verbosity.

#### `DetrLearner.load`
```python
DetrLearner.load(self, path)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.

#### `DetrLearner.optimize`
```python
DetrLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:

- **do_constant_folding**: *bool, default=False*\
  ONNX format optimization.
  If *True*, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the ops that have all constant inputs, with pre-computed constant nodes.

#### `DetrLearner.download`
```python
DetrLearner.download(self, path, mode, verbose)
```

Download utility for various DETR components.
Downloads files depending on mode and saves them in the path provided.
It supports downloading:
  1. The default *resnet50* and *resnet101* pretrained models.
  2. The weights for *resnet50* and *resnet101* bacbones.
  3. A test dataset with a single COCO image and its annotation.

Parameters:

- **path** : *str, default=None.*\
  Local path to save the files.
- **mode** : *{'pretrained', 'weights', 'test_data'}, default='pretrained'*\
  This *str* determines what file to download.
  Note that for modes *'weights'* and *'pretrained'* a model is downloaded and loaded according to the value of
  *self.backbone*.
  Backbones for which pretrained models are available, are: *'resnet50'* and *'resnet101'*.
  Also, a pretrained model with dilation is downloaded in case *self.args.dilation* is *True*.
  In case *self.panoptic_segmentation* is *True*, a model that was pretrained on the COCO panoptic dataset is
  downloaded.
- **verbose** : *bool, default=True*\
  Enables the maximum verbosity.

#### ROS Node

A [ROS node](../../projects/opendr_ws/src/perception/scripts/object_detection_2d_detr.py) is available for performing
inference on an image stream.
Documentation on how to use this node can be found [here](../../projects/opendr_ws/src/perception/README.md).

#### Tutorials and Demos

A tutorial on performing inference is available
[here](../../projects/python/perception/object_detection_2d/detr/inference_tutorial.ipynb).
Furthermore, demos on performing [training](../../projects/python/perception/object_detection_2d/detr/train_demo.py),
[evaluation](../../projects/python/perception/object_detection_2d/detr/eval_demo.py) and
[inference](../../projects/python/perception/object_detection_2d/detr/inference_demo.py) are also available.



#### Examples

* **Training example using an `ExternalDataset`.**

  To train properly, the backbone weights are downloaded automatically in the `temp_path`.
  Default backbone is *'resnet50'*.
  The training and evaluation dataset should be present in the path provided, along with the JSON annotation files.
  The default COCO 2017 training data can be found [here](https://cocodataset.org/#download) (train, val, annotations).
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  from opendr.perception.object_detection_2d import DetrLearner
  from opendr.engine.datasets import ExternalDataset

  detr_learner = DetrLearner(temp_path='./parent_dir', batch_size=8, device="cuda")

  training_dataset = ExternalDataset(path="./data", dataset_type="COCO")
  validation_dataset = ExternalDataset(path="./data", dataset_type="COCO")

  detr_learner.fit(dataset=training_dataset, val_dataset=validation_dataset, logging_path="./logs")
  detr_learner.save('./saved_models/trained_model')
  ```
* **Training example with a custom `DatasetIterator`.**

  This example serves to show how a custom dataset can be created by a user and used for training.
  In this way, the user can easily train on its own dataset.
  In order to do this, the user should create a `DatasetIterator` object that outputs `(Image, BoundingBoxList)` tuples.
  Here we show an example for doing this for the COCO dataset, but this can be done for any dataset as long as the
  `DatasetIterator` outputs `(Image, BoundingBoxList)` tuples.

  ```python
  import os
  import numpy as np
  from pycocotools.coco import COCO
  from opendr.engine.datasets import DatasetIterator
  from opendr.engine.data import Image
  from opendr.engine.target import BoundingBoxList
  from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner
  from PIL import Image as im

  # We create a DatasetIterator object that loads coco images and annotations and outputs (Image, BoundingBoxList) tuples.
  class CocoDatasetIterator(DatasetIterator):
      def __init__(self, image_folder, annotations_file):
          super().__init__()
          self.root = os.path.expanduser(image_folder)
          self.coco = COCO(annotations_file)
          self.ids = list(self.coco.imgs.keys())

      def __getitem__(self, idx):
          # Get ids of image and annotations
          img_id = self.ids[idx]
          ann_ids = self.coco.getAnnIds(imgIds=img_id)

          # Load the annotations with pycocotools
          target = self.coco.loadAnns(ann_ids)

          # Convert coco annotations to BoundingBoxList objects
          bounding_box_list = BoundingBoxList.from_coco(target, image_id=img_id)

          # Load images
          path = self.coco.loadImgs(img_id)[0]['file_name']
          img = im.open(os.path.join(self.root, path)).convert('RGB')

          # Convert image to Image object
          image = Image(np.array(img))

          return image, bounding_box_list

      def __len__(self):
          return len(self.ids)

  # We create a learner that trains for 3 epochs
  learner = DetrLearner(iters=3, temp_path="temp")

  # We download a pretrained detr model from the detr repo
  learner.download()

  # Download dummy dataset with a single picture
  learner.download("test_data")

  # The dummy dataset is stored in the temp_path
  image_folder = "temp/nano_coco/image"
  annotations_file = "temp/nano_coco/instances.json"

  dataset = CocoDatasetIterator(image_folder, annotations_file)

  learner.fit(dataset)
  ```

* **Inference and result drawing example on a test .jpg image, similar to and partially copied from [detr_demo colab](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=Jf59UNQ37QhJ).**

  This example shows how to perform inference on an image and draw the resulting bounding boxes using a detr model that is
  pretrained on the coco dataset.

  ```python
  import numpy as np
  import urllib
  import cv2
  from opendr.perception.object_detection_2d import DetrLearner
  from opendr.perception.object_detection_2d.detr.algorithm.util.draw import draw


  # Download an image
  url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
  req = urllib.request.urlopen(url)
  arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
  img = cv2.imdecode(arr, -1)

  learner = DetrLearner(threshold=0.7, backbone='resnet101')
  learner.download()
  bounding_box_list = learner.infer(img)
  cv2.imshow('Detections', draw(img, bounding_box_list))
  cv2.waitKey(0)
  ```

* **Inference and result drawing example on a test .jpg image with segmentations, similar to [detr_demo colab](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb#scrollTo=LAjJjP9kAHhA).**

  This example shows how to perform inference on an image and draw the resulting bounding boxes and segmentations using a detr
  model that is pretrained on the coco_panoptic dataset.

  ```python
  import numpy as np
  import urllib
  import cv2
  from opendr.perception.object_detection_2d import DetrLearner
  from opendr.perception.object_detection_2d.detr.algorithm.util.draw import draw

  # Download an image
  url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
  req = urllib.request.urlopen(url)
  arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
  img = cv2.imdecode(arr, -1)

  # We want to return the segmentations and plot those, so we set panoptic_segmentation to True.
  # Also, we have to modify the number of classes, since the number of panoptic classes in the pretrained detr model is 250.
  learner = DetrLearner(panoptic_segmentation=True, num_classes=250)
  learner.download()
  bounding_box_list = learner.infer(img)
  cv2.imshow('Detections', draw(img, bounding_box_list))
  cv2.waitKey(0)
  ```

* **Optimization example for a previously trained model.**

  Inference can be run with the trained model after running self.optimize.

  ```python
  from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner

  detr_learner = DetrLearner()
  detr_learner.download()
  detr_learner.optimize()
  detr_learner.save('./parent_dir/optimized_model')
  ```


#### References
<a name="detr-paper" href="https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers">[1]</a>
End-to-end Object Detection with Transformers,
[arXiv](https://arxiv.org/abs/2005.12872).
