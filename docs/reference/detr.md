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
DetrLearner(model_config_path, iters, lr, batch_size, optimizer, backbone, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, threshold)
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

#### `DetrLearner.download_model`
```python
DetrLearner.download_model(self, panoptic, backbone, dilation, pretrained)
```

Method for downloading (pretrained) models from the [DETR github](https://github.com/facebookresearch/detr).

Parameters:
- **panoptic** : *bool, default=False.*
  This bool differentiates between coco or coco_panoptic models. If False, a coco model is downloaded instead of a coco_panoptic model.
- **backbone** : *str, default='resnet50'*
  This str determines the backbone that is used in the model. There are two possible backbones: "resnet50" and "resnet101".
- **dilation** : *bool, default=False*
  If set to true, dilation is used in the model, otherwise not.
- **pretrained** : *bool, default=True*
  If True, a pretrained model is downloaded.

  #### `DetrLearner.download_nano_coco`
  ```python
  DetrLearner.download_nano_coco(self)
  ```
Method for downloading a minimal coco dataset from the OpenDR server that contains a single image for running tests. The dataset will be saved in the `temp_path`.

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
  
  detr_learner.fit(dataset=training_dataset, val_dataset=validation_dataset, logging_path="./logs)
  detr_learner.save('./saved_models/trained_model')
  ```
* **Training example with a custom `DatasetIterator`.**
  This example serves to show how a custom dataset can be created by a user and used for training. 
  This can be used for cases where the user does not want to make use of the dataset type `ExternalDataset` or does not want to use the standard input data type or target data type. 
  We create a custom dataset using the `DatasetIterator` class and create appropriate transformations using the `MappedDatasetIterator` class.
  The `DatasetIterator` we create outputs tuples of type `(Image, BoundingBoxList)`. 
  Since the DETR algorithm expects its own data format, we create a mapping function to allow the learner to work with this dataset.
  
    ```python
    import os
    import numpy as np
    from pycocotools.coco import COCO
    from engine.datasets import MappedDatasetIterator, DatasetIterator
    from engine.data import Image
    from engine.target import BoundingBoxList
    from perception.object_detection_2d.detr.detr_learner import DetrLearner
    from perception.object_detection_2d.detr.algorithm.datasets.coco import (
        ConvertCocoPolysToMask, make_coco_transforms)
    from PIL import Image as im
    
    # Create mapping function to convert (Image, BoundingBoxList) to detr format
    def create_map_bounding_box_list_dataset(image_set, return_masks):
                
                prepare = ConvertCocoPolysToMask(return_masks)
                transforms = make_coco_transforms(image_set)
                
                def map(data):
                    image, target = data
                    numpy_image = image.numpy()
                    pil_image = im.fromarray(numpy_image)
                    
                    coco_target = {'image_id' : target.image_id, 'annotations' : target.coco()}
                    image, target = prepare(pil_image, coco_target)
                    transformed_img, transformed_target = transforms(image, target)
                    return transformed_img, transformed_target
    
                return map
    
    # Create DatasetIterator object for Coco data
    class CocoDatasetIterator(DatasetIterator):
        def __init__(
            self, image_folder, annotations_file
            ):
    
            super().__init__()
            self.root = os.path.expanduser(image_folder)
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.imgs.keys())
    
        def __getitem__(self, idx):
            coco = self.coco
            img_id = self.ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            bounding_box_list = BoundingBoxList.from_coco(target, image_id=img_id)
            path = coco.loadImgs(img_id)[0]['file_name']
            img = im.open(os.path.join(self.root, path)).convert('RGB')
            image = Image(np.array(img))
            return image, bounding_box_list
    
        def __len__(self):
            return len(self.ids)
    
        def __repr__(self):
            fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
            fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
            fmt_str += '    Root Location: {}\n'.format(self.root)
            return fmt_str
    
    learner = DetrLearner()
    learner.download_model()
    
    # Download dummy dataset
    learner.download_nano_coco()
    
    image_folder = "temp/nano_coco/image"
    annotations_file = "temp/nano_coco/instances.json"
    
    dataset = CocoDatasetIterator(image_folder, annotations_file)
    map = create_map_bounding_box_list_dataset("train", False)
    
    mapped_dataset = MappedDatasetIterator(dataset, map)
    
    learner.fit(mapped_dataset)
    ```


* **Inference and result drawing example on a test .jpg image, similar to the [detr_demo colab](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=Jf59UNQ37QhJ).**
    ```python
    from perception.object_detection_2d.detr.detr_learner import DetrLearner
    from PIL import Image
    import matplotlib.pyplot as plt
    import requests
    
    # Function for plotting results
    def plot_results(pil_img, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for box in boxes:
        ax.add_patch(plt.Rectangle((box.left, box.top), box.width, box.height,
                                   fill=False, linewidth=3))
        text = f'{box.name}: {box.confidence:0.2f}'
        ax.text(box.left, box.top, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
    
    # Download an image
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    img = Image.open(requests.get(url, stream=True).raw)
    
    detr_learner = DetrLearner()
    detr_learner.download_model()
    bounding_box_list = detr_learner.infer(img)
    plot_results(img, bounding_box_list)
    ```

* **Optimization example for a previously trained model.**
  Inference can be run with the trained model after running self.optimize.
  ```python
  from perception.object_detection_2d.detr.detr_learner import DetrLearner

  detr_learner = DetrLearner()
  detr_learner.download_model()  
  detr_learner.optimize()
  detr_learner.save('./parent_dir/optimized_model')
  ```


#### References
<a name="detr-paper" href="https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers">[1]</a> End-to-end Object Detection with Transformers,
[arXiv](https://arxiv.org/abs/2005.12872).  
