## gesture_recognition module

The gesture recognition module contains the GestureRecognitionLearner class and can be used to recognize and localize 18 hand gestures. The module relies on nanodet object detection module. We provide data processing scripts and a pre-trained model for [Hagrid dataset](https://github.com/hukenovs/hagrid/tree/master).

### Class GestureRecognitionLearner
Bases: `object_detection_2d.nanodet.NanodetLearner`

The learner has the following public methods:

#### `GestureRecognitionLearner` constructor
```python
GestureRecognitionLearner(self, model_to_use, iters, lr, batch_size, checkpoint_after_iter, checkpoint_load_iter, temp_path, device,
               weight_decay, warmup_steps, warmup_ratio, lr_schedule_T_max, lr_schedule_eta_min, grad_clip)
```

Constructor parameters:

- **model_to_use**: *{"plus_m_1.5x_416"}, default=plus_m_1.5x_416*\
  Specifies the model to use and the config file. Currently plus_m_1.5x_416 is supported, while other models can be created following the config file.
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

#### `GestureRecognitionLearner.preprocess_data`
```python
GestureRecognitionLearner.preprocess_data(self, preprocess, download, verbose, save_path)
```

This method is used for downloading the [gesture recognition dataset](https://github.com/hukenovs/hagrid/tree/master) and preprocessing it to COCO format.

Parameters:

- **preprocess**: *bool, default=True*\
  Indicates whether to preprocess data located in save_path to COCO format.
- **download** : *bool, default=False*\
  Indicates whether to download data to save_path.
- **logging_path** : *str, default=''*\
  Subdirectory in temp_path to save log files and TensorBoard.
- **verbose** : *bool, default=True*\
  Enables verbosity.
- **save_path** : *str, default='./data'*\
  Path where to save data or where the downloaded data that needs to be preprocessed is located.

#### `GestureRecognitionLearner.fit`
```python
GestureRecognitionLearner.fit(self, dataset, val_dataset, logging_path, verbose, logging, seed, local_rank)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset of `ExternalDataset` type.
- **val_dataset** : *object, default=None*\
  Object that holds the validation dataset of `ExternalDataset` type.
- **logging_path** : *str, default=''*\
  Subdirectory in temp_path to save log files and TensorBoard.
- **verbose** : *bool, default=True*\
  Enables verbosity.
- **logging** : *bool, default=False*\
  Enables the maximum verbosity and the logger.
- **seed** : *int, default=123*\
  Seed for repeatability.
- **local_rank** : *int, default=1*\
  Needed if training on multiple machines.

#### `GestureRecognitionLearner.eval`
```python
GestureRecognitionLearner.eval(self, dataset, verbose, logging, local_rank)
```

This method is used to evaluate a trained model on an evaluation dataset.
Saves a txt logger file containing stats regarding evaluation.

Parameters:

- **dataset** : *object*\
  Object that holds the evaluation dataset of type `ExternalDataset`.
- **verbose**: *bool, default=True*\
  Enables verbosity.
- **logging**: *bool, default=False*\
  Enables the maximum verbosity and logger.
- **local_rank** : *int, default=1*\
  Needed if evaluating on multiple machines.

#### `GestureRecognitionLearner.infer`
```python
GestureRecognitionLearner.infer(self, input, conf_threshold, iou_threshold, nms_max_num)
```

This method is used to perform gesture recognition (detection) on an image.
Returns an `engine.target.BoundingBoxList` object, which contains bounding boxes that are described by the top-left corner and
their width and height, or returns an empty list if no detections were made on the input image.

Parameters:
- **input** : *object*\
  Object of type engine.data.Image.
  Image type object to perform inference on.
- **conf_threshold**: *float, default=0.35*\
  Specifies the threshold for gesture detection inference.
  An object is detected if the confidence of the output is higher than the specified threshold.
- **iou_threshold**: *float, default=0.6*\
  Specifies the IOU threshold for NMS in inference.
- **nms_max_num**: *int, default=100*\
  Determines the maximum number of bounding boxes that will be retained following the nms.

#### `GestureRecognitionLearner.save`
```python
GestureRecognitionLearner.save(self, path, verbose)
```

This method is used to save a trained model with its metadata.
Provided with the path, it creates the *path* directory, if it does not already exist.
Inside this folder, the model is saved as *nanodet_{model_name}.pth* and a metadata file *nanodet_{model_name}.json*.
If the directory already exists, the *nanodet_{model_name}.pth* and *nanodet_{model_name}.json* files are overwritten.
If optimization is performed, the optimized model is saved instead.

Parameters:

- **path**: *str, default=None*\
  Path to save the model, if None it will be `"temp_folder"` or `"cfg.save_dir"` from the learner.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity and logger.

#### `GestureRecognitionLearner.load`
```python
GestureRecognitionLearner.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.
If optimization is performed, the optimized model is loaded instead.

Parameters:

- **path**: *str, default=None*\
  Path of the model to be loaded.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity.

#### `GestureRecognitionLearner.download`
```python
GestureRecognitionLearner.download(self, path, model, verbose, url)
```

Downloads the provided pretrained model.

Parameters:

- **path**: *str, default=None*\
  Specifies the folder where data will be downloaded. If *None*, the *self.temp_path* directory is used instead.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.

#### Examples

* **Training example**

  ```python
  from opendr.perception.gesture_recognition.gesture_recognition_learner import GestureRecognitionLearner


  if __name__ == '__main__':
    model_save_dir = './save_dir/'
    data_save_dir = './data/'

    gesture_model = GestureRecognitionLearner(model_to_use='plus_m_1.5x_416', iters=100, lr=1e-3, batch_size=32,checkpoint_after_iter=1, checkpoint_load_iter=0, device="cuda", temp_path = model_save_dir)

    dataset, val_dataset, test_dataset = gesture_model.preprocess_data(preprocess=True, download=True, verbose=True, save_path=data_save_dir)

    gesture_model.fit(dataset, val_dataset, logging_path = './logs', logging=True)
    gesture_model.save()

  ```

* **Inference and result drawing example on a test image**

  This example shows how to perform inference on an image and draw the resulting bounding boxes

  ```python
  from opendr.perception.gesture_recognition.gesture_recognition_learner import GestureRecognitionLearner
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import draw_bounding_boxes

  if __name__ == '__main__':
    gesture_model = GestureRecognitionLearner(model_to_use='plus_m_1.5x_416')
    gesture_model.download("./")
    gesture_model.load("./nanodet_plus_m_1.5x_416", verbose=True)
    img = Image.open("./test_image.jpg")
    boxes = gesture_model.infer(input=img)

    draw_bounding_boxes(img.opencv(), boxes, class_names=gesture_model.classes, show=True)
  ```

