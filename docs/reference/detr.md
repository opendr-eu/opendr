## detr module

The *detr* module contains the *DetrLearner* class, which inherits from the abstract class *Learner*.

### Class DetrLearner
Bases: `engine.learners.Learner`

The *DetrLearner* class is a wrapper of the DETR [[1]](#detr-paper) object detection algorithm based on the original [DETR implementation](https://github.com/facebookresearch/detr).
It can be used to perform object detection on images (inference) and train DETR object detection models.

The [DetrLearner](#src.perception.object_detection_2d.detr.detr_learner.py) class has the
following public methods:

#### `DetrLearner` constructor
```python
DetrLearner(model_config_path, iters, lr, batch_size, optimizer, backbone, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, threshold, num_classes, return_segmentations)
```

Constructor parameters:
- **model_config_path**: *str, default="OpenDR/src/perception/object_detection_2d/detr/algorithm/config/model_config.yaml"*
  Specifies the path to the config file that contains the additional parameters from the original [DETR implementation](https://github.com/facebookresearch/detr).
- **iters**: *int, default=10*
  Specifies the number of epochs the training should run for.
- **lr**: *float, default=1e-4*
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=1*
  Specifies number of images to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **optimizer**: *{'sdg', 'adam', 'adamw'}, default='adamw'*
  Specifies the type of optimizer that is used during training.
- **backbone**: *{'resnet50', 'resnet101'}, default='resnet50'*
  Specifies the backbone architecture. Other Torchvision backbones are also valid, but have no pretrained DETR models available. Therefore other backbone models have to be learned from scratch.
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
  Specifies the number of classes of the model. The default is 91, since this is the number of classes in the COCO dataset, but modifying the num_classes allows the user to train on its own dataset.
  It is also possible to use pretrained DETR models with the specified num_classes, since the head of the pretrained model with be modified appropriately.
  In this way, a model that was pretrained on the coco dataset can be finetuned to another dataset. Training on other datasets than COCO can be done by creating a DatasetIterator that outputs (Image, BoundingBoxList) tuples.
  Below you can find an example that shows how you can create such a DatasetIterator.
- **return_segmentations**: *bool, default=False*
  Specifies whether the model returns, next to bounding boxes, segmentations of objects. If True, the `download_model()` method will download pretrained coco_panoptic models.

#### `DetrLearner.fit`
```python
DetrLearner.fit(dataset, val_dataset, logging_path, silent, verbose, annotations_folder, train_images_folder, train_annotations_file, val_images_folder, val_annotations_file)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Returns a dictionary containing stats regarding the last evaluation ran.
Parameters:
- **dataset**: *object*
    Object that holds the training dataset. Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **val_dataset** : *object, default=None*
    Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`. Object that holds the validation dataset.
- **logging_path** : *str, default=''*
    Path to save tensorboard log files. If set to None or '', tensorboard logging is disabled.
- **silent** : *bool, default=False*
    If True, all printing of training progress reports and other information to STDOUT are disabled.
- **verbose** : *bool, default=True*
    Enables the maximum verbosity.
- **annotations_folder** : *str, default='Annotations'*
    Foldername of the annotations json file. This folder should be contained in the dataset path provided.
- **train_images_folder** : *str, default='train2017'*
    Name of the folder that contains the train dataset images. This folder should be contained in the dataset path provided. Note that this is a folder name, not a path.
- **train_annotations_file** : *str, default='instances_train2017.json'*
    Filename of the train annotations json file. This file should be contained in the dataset path provided.
-  **val_images_folder** : *str, default='val2017'*
    Folder name that contains the validation images. This folder should be contained in the dataset path provided. Note that this is a folder name, not a path.
- **val_annotations_file** : *str, default='instances_val2017.json'*
    Filename of the validation annotations json file. This file should be contained in the dataset path provided in the annotations folder provided.

#### `DetrLearner.eval`
```python
DetrLearner.eval(self, dataset, images_folder, annotations_folder, annotations_file)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.
Parameters:
- **dataset** : *object*
    ExternalDataset class object or DatasetIterator class object. Object that holds the evaluation dataset.
- **images_folder** : *str, default='val2017'*
    Folder name that contains the dataset images. This folder should be contained in the dataset path provided. Note that this is a folder name, not a path.
- **annotations_folder** : *str, default='Annotations'*
    Folder name of the annotations json file. This file should be contained in the dataset path provided.
- **annotations_file** : *str, default='instances_val2017.json'*
    Filename of the annotations json file. This file should be contained in the dataset path provided.

#### `DetrLearner.infer`
```python
DetrLearner.infer(image)
```

This method is used to perform object detection on an image.
Returns an engine.target.BoundingBoxList object, which contains bounding boxes that are described by the left-top corner and its width and height, or returns an empty list if no detections were made.

Parameters:
- **image : *object*
    Image of type engine.data.Image class or np.array. Image to run inference on.

#### `DetrLearner.save`
```python
DetrLearner.save(self, path)
```

This method is used to save a trained model.
Provided with the path, it creates the "name" directory, if it does not already
exist. Inside this folder, the model is saved as "detr_[backbone_model].pth" and the metadata file as "detr_[backbone].json". If the directory
already exists, the "detr_[backbone_model].pth" and "detr_[backbone].json" files are overwritten.

If [`self.optimize`](#DetrLearner.optimize) was run previously, it saves the optimized ONNX model in a similar fashion with an ".onnx" extension, by copying it from the self.temp_path it was saved previously during conversion.

Parameters:
- **path**: *str*
  Path to save the model, including the filename.

#### `DetrLearner.load`
```python
DetrLearner.load(self, path)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:
- **path**: *str*
  Path of the model to be loaded.

#### `DetrLearner.optimize`
```python
DetrLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export. Constant-folding optimization will replace some of the ops that have all constant inputs, with pre-computed constant nodes.

#### `DetrLearner.download`
```python
DetrLearner.download(self, panoptic, backbone, dilation, pretrained)
```

Download utility for various DETR components. Downloads files depending on mode and
        saves them in the path provided. It supports downloading:
        1) the default resnet50 and resnet101 pretrained models
        2) resnet50 and resnet101 weights needed for training
        3) a test dataset with a single COCO image and its annotation

Parameters:
- **path** : *str, default=None.*
  Local path to save the files.
- **mode** : *str, default='pretrained'*
  This str determines the backbone that is used in the model. There are two possible backbones: "resnet50" and "resnet101".
- **verbose** : *bool, default=False*
  If set to true, dilation is used in the model, otherwise not.
- **pretrained** : *bool, default=True*
  Whether to print all output in the console. The default is False.

#### Examples

* **Training example using an `ExternalDataset`.**
  To train properly, the backbone weights are downloaded automatically in the `temp_path`. Default backbone is
  'resnet50'.
  The training and evaluation dataset should be present in the path provided, along with the JSON annotation files.
  The default COCO 2017 training data can be found [here](https://cocodataset.org/#download) (train, val, annotations).
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  from OpenDR.perception.object_detection_2d.detr.detr_learner import DetrLearner
  from OpenDR.engine.datasets import ExternalDataset

  detr_learner = DetrLearner(temp_path='./parent_dir', batch_size=8, device="cuda")

  training_dataset = ExternalDataset(path="./data", dataset_type="COCO")
  validation_dataset = ExternalDataset(path="./data", dataset_type="COCO")

  detr_learner.fit(dataset=training_dataset, val_dataset=validation_dataset, logging_path="./logs")
  detr_learner.save('./saved_models/trained_model')
  ```
* **Training example with a custom `DatasetIterator`.**
  This example serves to show how a custom dataset can be created by a user and used for training.
  In this way, the user can easily train on its own dataset. In order to do this, the user should create a `DatasetIterator` object that outputs `(Image, BoundingboxList)` tuples.
  Here we show an example for doing this for the COCO dataset, but this can be done for any dataset as long as the `DatasetIterator` outputs `(Image, BoundingboxList)` tuples.

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
	This example shows how to perform inference on an image and draw the resulting bounding boxes using a detr model that is pretrained on the coco dataset.
    ```python
    import matplotlib.pyplot as plt
    import requests
    from PIL import Image as im
    from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner

    # COCO classes
    classes = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

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
            text = f'{classes[box.name]}: {box.confidence:0.2f}'
            ax.text(box.left, box.top, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

    # Download an image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    img = im.open(requests.get(url, stream=True).raw)

    learner = DetrLearner(threshold=0.7)
    learner.download()
    bounding_box_list = learner.infer(img)
    plot_results(img, bounding_box_list)
    ```

* **Inference and result drawing example on a test .jpg image with segmentations, similar to [detr_demo colab](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb#scrollTo=LAjJjP9kAHhA).**
	This example shows how to perform inference on an image and draw the resulting bounding boxes and segmentations using a detr model that is pretrained on the coco_panoptic dataset.

    ```python
	import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import requests
    from PIL import Image as im
    import opendr
    from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner

    # These are the COCO classes
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]


    # Function for plotting results
    def plot_results(pil_img, boxes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        color = []
        polygons = []
        for box in boxes:
            if box.name >= len(CLASSES):
                continue
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            ax.add_patch(plt.Rectangle((box.left, box.top), box.width, box.height,
                                       fill=False, linewidth=3))
            text = f'{CLASSES[box.name]}: {box.confidence:0.2f}'
            seg = box.segmentation[0]
            poly = np.array(seg).reshape((int(len(seg)/2), 2))
            polygons.append(Polygon(poly))
            color.append(c)

            ax.text(box.left, box.top, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        plt.axis('off')
        plt.show()

    # Download an image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    img = im.open(requests.get(url, stream=True).raw)

    # We want to return the segmentations and plot those, so we set return_segmentations to True.
    # Also, we have to modify the number of classes, since the number of panoptic classes in the pretrained detr model is 250.
    learner = DetrLearner(return_segmentations=True, num_classes=250)
    learner.download()
    bounding_box_list = learner.infer(img)
    plot_results(img, bounding_box_list)
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
<a name="detr-paper" href="https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers">[1]</a> End-to-end Object Detection with Transformers,
[arXiv](https://arxiv.org/abs/2005.12872).  
