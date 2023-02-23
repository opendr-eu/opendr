## nanodet module

The *nanodet* module contains the *NanodetLearner* class, which inherits from the abstract class *Learner*.

### Class NanodetLearner
Bases: `engine.learners.Learner`

The *NanodetLearner* class is a wrapper of the Nanodet object detection algorithms based on the original
[Nanodet implementation](https://github.com/RangiLyu/nanodet).
It can be used to perform object detection on images (inference) and train all predefined Nanodet object detection models and new modular models from the user.

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
  "plus_m_1.5x_416", "custom"}, default=m*\
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
NanodetLearner.fit(self, dataset, val_dataset, logging_path, verbose, logging, seed, local_rank)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset.
  Can be of type `ExternalDataset` or `XMLBasedDataset`.
- **val_dataset** : *object, default=None*\
  Object that holds the validation dataset.
  Can be of type `ExternalDataset` or `XMLBasedDataset`.
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

#### `NanodetLearner.eval`
```python
NanodetLearner.eval(self, dataset, verbose, logging, local_rank)
```

This method is used to evaluate a trained model on an evaluation dataset.
Saves a txt logger file containing stats regarding evaluation.

Parameters:

- **dataset** : *object*\
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or `XMLBasedDataset`.
- **verbose**: *bool, default=True*\
  Enables verbosity.
- **logging**: *bool, default=False*\
  Enables the maximum verbosity and logger.
- **local_rank** : *int, default=1*\
  Needed if evaluating on multiple machines.

#### `NanodetLearner.infer`
```python
NanodetLearner.infer(self, input, conf_threshold, iou_threshold, nms_max_num)
```

This method is used to perform object detection on an image.
Returns an `engine.target.BoundingBoxList` object, which contains bounding boxes that are described by the top-left corner and
their width and height, or returns an empty list if no detections were made on the input image.

Parameters:
- **input** : *object*\
  Object of type engine.data.Image.
  Image type object to perform inference on.
- **conf_threshold**: *float, default=0.35*\
  Specifies the threshold for object detection inference.
  An object is detected if the confidence of the output is higher than the specified threshold.
- **iou_threshold**: *float, default=0.6*\
  Specifies the IOU threshold for NMS in inference.
- **nms_max_num**: *int, default=100*\
  Determines the maximum number of bounding boxes that will be retained following the nms.

#### `NanodetLearner.optimize`
```python
NanodetLearner.optimize(self, export_path, verbose, optimization, conf_threshold, iou_threshold, nms_max_num)
```

This method is used to perform JIT or ONNX optimizations and save a trained model with its metadata.
If a model is not present in the location specified by *export_path*, the optimizer will save it there.
If a model is already present, it will load it instead.
Inside this folder, the model is saved as *nanodet_{model_name}.pth* for JIT models or *nanodet_{model_name}.onnx* for ONNX and a metadata file *nanodet_{model_name}.json*.

Note: In ONNX optimization, the output model executes the original model's feed forward method.
The user must create their own pre- and post-processes in order to use the ONNX model in the C API.
In JIT optimization the output model performs the feed forward pass and post-processing.
To use the C API, it is recommended to use JIT optimization as shown in the [example of OpenDR's C API](../../projects/c_api/samples/object_detection/nanodet/nanodet_jit_demo.c).

Parameters:

- **export_path**: *str*\
  Path to save or load the optimized model.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity.
- **optimization**: *str, default="jit"*\
  It determines what kind of optimization is used, possible values are *jit* or *onnx*.
- **conf_threshold**: *float, default=0.35*\
  Specifies the threshold for object detection inference.
  An object is detected if the confidence of the output is higher than the specified threshold.
- **iou_threshold**: *float, default=0.6*\
  Specifies the IOU threshold for NMS in inference.
- **nms_max_num**: *int, default=100*\
  Determines the maximum number of bounding boxes that will be retained following the nms.

#### `NanodetLearner.save`
```python
NanodetLearner.save(self, path, verbose)
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

#### `NanodetLearner.load`
```python
NanodetLearner.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.
If optimization is performed, the optimized model is loaded instead.

Parameters:

- **path**: *str, default=None*\
  Path of the model to be loaded.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity.

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
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.


#### Tutorials and Demos

A Jupyter notebook tutorial on performing inference is [available](../../projects/python/perception/object_detection_2d/nanodet/inference_tutorial.ipynb).
Furthermore, demos on performing [training](../../projects/python/perception/object_detection_2d/nanodet/train_demo.py),
[evaluation](../../projects/python/perception/object_detection_2d/nanodet/eval_demo.py) and
[inference](../../projects/python/perception/object_detection_2d/nanodet/inference_demo.py) are also available.



#### Examples

* **Training example using an `ExternalDataset`**

  To train properly, the architecture weights must be downloaded in a predefined directory before fit is called, in this case the directory name is "predefined_examples".
  Default architecture is *'m'*.
  The training and evaluation dataset root should be present in the path provided, along with the annotation files.
  The default COCO 2017 training data can be found [here](https://cocodataset.org/#download) (train, val, annotations).
  All training parameters (optimizer, lr schedule, losses, model parameters etc.) can be changed in the model config file
  in [config directory](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/config).
  You can find more information in [corresponding documentation](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/config/config_file_detail.md).
  For easier usage of the NanodetLearner, you can overwrite the following parameters:
  (iters, lr, batch_size, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, weight_decay, warmup_steps,
  warmup_ratio, lr_schedule_T_max, lr_schedule_eta_min, grad_clip)

  **Note**

  The Nanodet tool can be used with any PASCAL VOC- or COCO-like dataset, by providing the correct root and dataset type.

  If *'voc'* is chosen for *dataset*, the directory must look like this:

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

  On the other hand, if *'coco'* is chosen for *dataset*, the directory must look like this:

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

  You can change the default annotation and image directories in [the *build_dataset* function](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/nanodet/data/dataset/__init__.py).
  This example assumes the data has been downloaded and placed in the directory referenced by `data_root`.
  ```python
  from opendr.engine.datasets import ExternalDataset
  from opendr.perception.object_detection_2d import NanodetLearner


  if __name__ == '__main__':
    dataset = ExternalDataset(data_root, 'voc')
    val_dataset = ExternalDataset(data_root, 'voc')

    nanodet = NanodetLearner(model_to_use='m', iters=300, lr=5e-4, batch_size=8,
                             checkpoint_after_iter=50, checkpoint_load_iter=0,
                             device="cpu")

    nanodet.download("./predefined_examples", mode="pretrained")
    nanodet.load("./predefined_examples/nanodet_m", verbose=True)
    nanodet.fit(dataset, val_dataset)
    nanodet.save()

  ```

* **Inference and result drawing example on a test image**

  This example shows how to perform inference on an image and draw the resulting bounding boxes using a nanodet model that is pretrained on the COCO dataset.
  In this example, a pre-trained model is downloaded and inference is performed on an image that can be specified with the *path* parameter.

  ```python
  from opendr.perception.object_detection_2d import NanodetLearner
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import draw_bounding_boxes

  if __name__ == '__main__':
    nanodet = NanodetLearner(model_to_use='m', device="cpu")
    nanodet.download("./predefined_examples", mode="pretrained")
    nanodet.load("./predefined_examples/nanodet_m", verbose=True)
    nanodet.download("./predefined_examples", mode="images")
    img = Image.open("./predefined_examples/000000000036.jpg")
    boxes = nanodet.infer(input=img)

    draw_bounding_boxes(img.opencv(), boxes, class_names=nanodet.classes, show=True)
  ```

* **Optimization framework with Inference and result drawing example on a test image**

  This example shows how to perform optimization on a pretrained model, then run inference on an image and finally draw the resulting bounding boxes, using a nanodet model that is pretrained on the COCO dataset.
  In this example we use ONNX optimization, but JIT can also be used by changing *optimization* to *jit*.
  The optimized model will be saved in the `./optimization_models` folder
  ```python
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import NanodetLearner, draw_bounding_boxes


  if __name__ == '__main__':
    nanodet = NanodetLearner(model_to_use='m', device="cpu")
    nanodet.load("./predefined_examples/nanodet_m", verbose=True)

    # First read an OpenDR image from your dataset and run the optimizer:
    img = Image.open("./predefined_examples/000000000036.jpg")
    nanodet.optimize("./onnx/nanodet_m/", optimization="onnx")

    boxes = nanodet.infer(input=img)

    draw_bounding_boxes(img.opencv(), boxes, class_names=nanodet.classes, show=True)
  ```


#### Performance Evaluation

In terms of speed, the performance of Nanodet is summarized in the tables below (in FPS).
The speed is measured from the start of the forward pass until the end of post-processing.

For PyTorch inference:

| Method              {input} | RTX 2070 | TX2   | NX    |
|-----------------------------|----------|-------|-------|
| Efficient Lite0     {320}   | 81.98    | 15.51 | 22.75 |
| Efficient Lite1     {416}   | 60.09    | 11.27 | 19.19 |
| Efficient Lite2     {512}   | 59.46    | 8.53  | 15.99 |
| RepVGG A0           {416}   | 48.13    | 13.33 | 21.46 |
| Nanodet-g           {416}   | 89.93    | 15.59 | 21.67 |
| Nanodet-t           {320}   | 63.83    | 13.33 | 19.60 |
| Nanodet-m           {320}   | 67.90    | 13.38 | 19.36 |
| Nanodet-m 0.5x      {320}   | 69.69    | 12.69 | 18.84 |
| Nanodet-m 1.5x      {320}   | 65.77    | 13.95 | 18.45 |
| Nanodet-m           {416}   | 71.76    | 13.06 | 17.88 |
| Nanodet-m 1.5x      {416}   | 63.51    | 13.11 | 19.31 |
| Nanodet-plus m      {320}   | 52.32    | 11.32 | 17.99 |
| Nanodet-plus m 1.5x {320}   | 52.11    | 11.54 | 17.05 |
| Nanodet-plus m      {416}   | 59.25    | 11.48 | 17.14 |
| Nanodet-plus m 1.5x {416}   | 52.35    | 9.34  | 16.78 |

For JIT optimization inference:

| Method              {input} | RTX 2070 | TX2   | NX    |
|-----------------------------|----------|-------|-------|
| Efficient Lite0     {320}   | 108.64   | 18.56 | 27.39 |
| Efficient Lite1     {416}   | 96.63    | 12.49 | 21.53 |
| Efficient Lite2     {512}   | 97.97    | 9.35  | 16.91 |
| RepVGG A0           {416}   | 48.23    | 16.59 | 23.77 |
| Nanodet-g           {416}   | 96.01    | 19.78 | 27.37 |
| Nanodet-t           {320}   | 99.85    | 18.17 | 23.74 |
| Nanodet-m           {320}   | 103.78   | 19.27 | 24.24 |
| Nanodet-m 0.5x      {320}   | 90.24    | 18.31 | 23.30 |
| Nanodet-m 1.5x      {320}   | 104.82   | 19.29 | 23.16 |
| Nanodet-m           {416}   | 100.61   | 12.08 | 22.34 |
| Nanodet-m 1.5x      {416}   | 92.37    | 18.45 | 22.89 |
| Nanodet-plus m      {320}   | 75.52    | 16.70 | 23.12 |
| Nanodet-plus m 1.5x {320}   |  86.23   | 16.83 | 21.64 |
| Nanodet-plus m      {416}   | 96.01    | 16.78 | 21.28 |
| Nanodet-plus m 1.5x {416}   | 86.97    | 14.42 | 21.53 |

For ONNX optimization inference:

| Method              {input} | CPU    | TX2   | NX    |
|-----------------------------|--------|-------|-------|
| Efficient Lite0     {320}   | 51.1   | 10.15 | 11.34 |
| Efficient Lite1     {416}   | 36.60  | 5.84  | 5.99  |
| Efficient Lite2     {512}   | 28.76  | 4.23  | 3.93  |
| RepVGG A0           {416}   | 83.03  | 9.49  | 9.49  |
| Nanodet-g           {416}   | 97.11  | 8.87  | 14.61 |
| Nanodet-t           {320}   | 87.34  | 13.22 | 19.06 |
| Nanodet-m           {320}   | 101.83 | 15.54 | 19.36 |
| Nanodet-m 0.5x      {320}   | 123.60 | 16.89 | 24.44 |
| Nanodet-m 1.5x      {320}   | 88.39  | 13.35 | 18.32 |
| Nanodet-m           {416}   | 83.42  | 12.51 | 17.11 |
| Nanodet-m 1.5x      {416}   | 76.30  | 9.85  | 14.79 |
| Nanodet-plus m      {320}   | 51.39  | 12.06 | 15.48 |
| Nanodet-plus m 1.5x {320}   | 63.19  | 9.55  | 11.69 |
| Nanodet-plus m      {416}   | 64.18  | 9.63  | 11.34 |
| Nanodet-plus m 1.5x {416}   | 52.36  | 6.98  | 8.59  |
Note that in embedded systems the standard deviation is around 0.2 - 0.3 seconds in larger networks cases.

Finally, we measure the performance on the COCO dataset, using the corresponding metrics:

| Method              {input} | coco2017 mAP |
|-----------------------------|--------------|
| Efficient Lite0     {320}   | 24.4         |
| Efficient Lite1     {416}   | 29.2         |
| Efficient Lite2     {512}   | 32.4         |
| RepVGG A0           {416}   | 25.5         |
| Nanodet-g           {416}   | 22.7         |
| Nanodet-m           {320}   | 20.2         |
| Nanodet-m 0.5x      {320}   | 13.1         |
| Nanodet-m 1.5x      {320}   | 23.1         |
| Nanodet-m           {416}   | 23.5         |
| Nanodet-m 1.5x      {416}   | 26.6         |
| Nanodet-plus m      {320}   | 27.0         |
| Nanodet-plus m 1.5x {320}   | 29.9         |
| Nanodet-plus m      {416}   | 30.3         |
| Nanodet-plus m 1.5x {416}   | 34.1         |
 