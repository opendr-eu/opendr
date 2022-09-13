## nanodet module

The *nanodet* module contains the *NanodetLearner* class, which inherits from the abstract class *Learner*.

### Class NanodetLearner
Bases: `engine.learners.Learner`

The *NanodetLearner* class is a wrapper of the Nanodet object detection algorithms based on the original
[Nanodet implementation](https://github.com/RangiLyu/nanodet).
It can be used to perform object detection on images (inference) and train All predefined Nanodet object detection models and new modular models from the user.

The [NanodetLearner](../../src/opendr/perception/object_detection_2d/nanodet/nanodet_learner.py) class has the
following public methods:

#### `NanodetLearner` constructor
```python
NanodetLearner(self, model_to_use, iters, lr, batch_size, checkpoint_after_iter, checkpoint_load_iter, temp_path, device,
               weight_decay, warmup_steps, warmup_ratio, lr_schedule_T_max, lr_schedule_eta_min, grad_clip)
```

Constructor parameters:

- **model_to_use**: *{"EfficientNet_Lite0_320", "EfficientNet_Lite1_416", "EfficientNet_Lite2_512", "RepVGG_A0_416",
  "t", "g", "m", "m_416", "m_0.5x", "m_1.5x", "m_1.5x_416", "plus_m_320", "plus_m_1.5x_320", "plus_m_416",
  "plus_m_1.5x_416", "custom"}, default=plus_m_1.5x_416*\
  Specifies the model to use and the config file that contains all hyperparameters for training, evaluation and inference as the original
  [Nanodet implementation](https://github.com/RangiLyu/nanodet). If you want to overwrite some of the parameters you can
  put them as parameters in the learner.
- **iters**: *int, default=None*\
  Specifies the number of epochs the training should run for.
- **lr**: *float, default=None*\
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=None*\
  Specifies number of images to be bundled up in a batch during training.
  This heavily affects memory usage, adjust according to your system.
- **checkpoint_after_iter**: *int, default=None*\
  Specifies per how many training iterations a checkpoint should be saved.
  If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=None*\
  Specifies which checkpoint should be loaded.
  If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*\
  Specifies a path where the algorithm looks for saving the checkpoints along with the logging files. If *''* the `cfg.save_dir` will be used instead.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **weight_decay**: *float, default=None*\
- **warmup_steps**: *int, default=None*\
- **warmup_ratio**: *float, default=None*\
- **lr_schedule_T_max**: *int, default=None*\
- **lr_schedule_eta_min**: *float, default=None*\
- **grad_clip**: *int, default=None*\

#### `NanodetLearner.fit`
```python
NanodetLearner.fit(self, dataset, val_dataset, logging_path, verbose, seed)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:

- **dataset**: *ExternalDataset*\
  Object that holds the training dataset.
  Can be of type `ExternalDataset`.
- **val_dataset** : *ExternalDataset, default=None*\
  Object that holds the validation dataset.
  Can be of type `ExternalDataset`.
- **logging_path** : *str, default=''*\
  Subdirectory in temp_path to save log files and TensorBoard.
- **verbose** : *bool, default=True*\
  Enables the maximum verbosity and the logger.
- **seed** : *int, default=123*\
  Seed for repeatability.

#### `NanodetLearner.eval`
```python
NanodetLearner.eval(self, dataset, verbose)
```

This method is used to evaluate a trained model on an evaluation dataset.
Saves a txt logger file containing stats regarding evaluation.

Parameters:

- **dataset** : *ExternalDataset*\
  Object that holds the evaluation dataset.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity and logger.

#### `NanodetLearner.infer`
```python
NanodetLearner.infer(self, input, thershold, verbose)
```

This method is used to perform object detection on an image.
Returns an `engine.target.BoundingBoxList` object, which contains bounding boxes that are described by the left-top corner and
its width and height, or returns an empty list if no detections were made of the image in input.

Parameters:
- **input** : *Image*\
  Image type object to perform inference on it. 
  - **threshold**: *float, default=0.35*\
  Specifies the threshold for object detection inference.
  An object is detected if the confidence of the output is higher than the specified threshold.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity and logger.

#### `NanodetLearner.save`
```python
NanodetLearner.save(self, path, verbose)
```

This method is used to save a trained model with its metadata.
Provided with the path, it creates the "path" directory, if it does not already exist.
Inside this folder, the model is saved as *"nanodet_{model_name}.pth"* and a metadata file *"nanodet_{model_name}.json"*.
If the directory already exists, the *"nanodet_{model_name}.pth"* and *"nanodet_{model_name}.json"* files are overwritten.

Parameters:

- **path**: *str, default=None*\
  Path to save the model, if None it will be the `"temp_folder"` or the `"cfg.save_dir"` from learner.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity and logger.

#### `NanodetLearner.load`
```python
NanodetLearner.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:

- **path**: *str, default=None*\
  Path of the model to be loaded.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity and logger.

#### `NanodetLearner.download`
```python
NanodetLearner.download(self, path, mode, model, verbose, url)
```

Downloads data needed for the various functions of the learner, e.g., pretrained models as well as test data.

Parameters:

- **path**: *str, default=None*\
  Specifies the folder where data will be downloaded. If *None*, the *self.temp_path* directory is used instead.
- **mode**: *{'pretrained', 'images', 'test_data'}, default='pretrained'*\
  If *'pretrained'*, downloads a pretrained detector model from the *model_to_use* architecture which was chosen at learner initialization.
  If *'images'*, downloads an image to perform inference on. If *'test_data'* downloads a dummy dataset for testing purposes.
- **verbose**: *bool, default=False*\
  Enables the maximum verbosity and logger.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.


#### Tutorials and Demos

A tutorial on performing inference is available.
Furthermore, demos on performing [training](../../projects/perception/object_detection_2d/nanodet/train_demo.py),
[evaluation](../../projects/perception/object_detection_2d/nanodet/eval_demo.py) and
[inference](../../projects/perception/object_detection_2d/nanodet/inference_demo.py) are also available.



#### Examples

* **Training example using an `ExternalDataset`.**

  To train properly, the architecture weights must be downloaded in a predefined directory before fit is called, in this case the directory name is "predefined_examples".
  Default architecture is *'plus-m-1.5x_416'*.
  The training and evaluation dataset root should be present in the path provided, along with the annotation files.
  The default COCO 2017 training data can be found [here](https://cocodataset.org/#download) (train, val, annotations).
  All training parameters (optimizer, lr schedule, losses, model parameters etc.) can be changed in the model config file 
  in [config directori](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/config). 
  You can find more informations in [config file detail](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/config/config_file_detail.md).
  For easier use, with NanodetLearner parameters user can overwrite the following parameters:
  (iters, lr, batch_size, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, weight_decay, warmup_steps,
  warmup_ratio, lr_schedule_T_max, lr_schedule_eta_min, grad_clip)
  
  **Note**
  
  The Nanodet tool can be used with any PASCAL VOC or COCO like dataset. The only thing is needed is to provide the correct root and dataset type.
  
  If *'voc'* is choosed for *dataset* the directory must look like this:
  
  - root folder
    - train
      - Annotations
        - image1.xml
        - image2.xml
        - ...
      - JPEGImages
        - image1.jpg
        - image2.jpg
        - ...
    - val
      - Annotations
        - image1.xml
        - image2.xml
        - ...
      - JPEGImages
        - image1.jpg
        - image2.jpg
        - ...

  On the other hand if *'coco'* is choosed for *dataset* the directory must look like this:
  
  - root folder
    - train2017
      - image1.jpg
      - image2.jpg
      - ...
    - val2017
      - image1.jpg
      - image2.jpg
      - ...
    - annotations
      - instances_train2017.json 
      - instances_val2017.json
   
  You can change the default annotation and image directories in [dataset](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/nanodet/data/dataset/__init__.py)

  ```python
  import argparse

  from opendr.engine.datasets import ExternalDataset
  from opendr.perception.object_detection_2d import NanodetLearner
  
  
  if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument("--dataset", help="Dataset to train on", type=str, default="coco", choices=["voc", "coco"])
      parser.add_argument("--data-root", help="Dataset root folder", type=str)
      parser.add_argument("--model", help="Model that config file will be used", type=str)
      parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
      parser.add_argument("--batch-size", help="Batch size to use for training", type=int, default=6)
      parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=5e-4)
      parser.add_argument("--checkpoint-freq", help="Frequency in-between checkpoint saving and evaluations", type=int, default=50)
      parser.add_argument("--n-epochs", help="Number of total epochs", type=int, default=300)
      parser.add_argument("--resume-from", help="Epoch to load checkpoint file and resume training from", type=int, default=0)
  
      args = parser.parse_args()
  
      if args.dataset == 'voc':
          dataset = ExternalDataset(args.data_root, 'voc')
          val_dataset = ExternalDataset(args.data_root, 'voc')
      elif args.dataset == 'coco':
          dataset = ExternalDataset(args.data_root, 'coco')
          val_dataset = ExternalDataset(args.data_root, 'coco')
  
      nanodet = NanodetLearner(model_to_use=args.model, iters=args.n_epochs, lr=args.lr, batch_size=args.batch_size,
                               checkpoint_after_iter=args.checkpoint_freq, checkpoint_load_iter=args.resume_from,
                               device=args.device)
  
      nanodet.download("./predefined_examples", mode="pretrained")
      nanodet.load("./predefined_examples/nanodet-{}/nanodet-{}.ckpt".format(args.model, args.model), verbose=True)
      nanodet.fit(dataset, val_dataset)
      nanodet.save()
  ```
  
* **Inference and result drawing example on a test image.**

  This example shows how to perform inference on an image and draw the resulting bounding boxes using a nanodet model that is pretrained on the COCO dataset.
  Moreover, inference can be used in all images in a folder, frames of a video or a webcam feedback with the provided *mode*.
  In this example first is downloaded a pre-trained model as in training example and then an image to be inference.
  With the same *path* parameter you can choose a folder or a video file to be used as inference. Last but not least, if 'webcam' is
  used in *mode* the *camid* parameter of inference must be used to determine the webcam device in your machine.
  
  ```python
  import argparse
  from opendr.perception.object_detection_2d import NanodetLearner
  
  if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
      parser.add_argument("--model", help="Model that config file will be used", type=str)
      args = parser.parse_args()
  
      nanodet = NanodetLearner(model_to_use=args.model, device=args.device)
  
      nanodet.download("./predefined_examples", mode="pretrained")
      nanodet.load("./predefined_examples/nanodet-{}/nanodet-{}.ckpt".format(args.model, args.model), verbose=True)
      nanodet.download("./predefined_examples", mode="images")
      boxes = nanodet.infer(path="./predefined_examples/000000000036.jpg")
  ```