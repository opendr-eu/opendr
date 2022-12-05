## object_tracking_2d_deep_sort module

The *object_tracking_2d_deep_sort* module contains the *ObjectTracking2DDeepSortLearner* class, which inherits from the abstract class *Learner*.

### Class ObjectTracking2DDeepSortLearner
Bases: `engine.learners.Learner`

The *ObjectTracking2DDeepSortLearner* class is a wrapper of the Deep SORT[[1]](#object-tracking-2d-1) implementation found on [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)[[2]](#object-tracking-2d-2).
It can be used to perform 2d object tracking on images and train new models.

The [ObjectTracking2DDeepSortLearner](/src/opendr/perception/object_tracking_2d/fair_mot/object_tracking_2d_fair_mot_learner.py) class has the
following public methods:

#### `ObjectTracking2DDeepSortLearner` constructor
```python
ObjectTracking2DDeepSortLearner(self, lr, iters, batch_size, optimizer, lr_schedule, backbone, network_head, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, threshold, scale, lr_step, head_conv, ltrb, num_classes, reg_offset, gpus, num_workers, mse_loss, reg_loss, dense_wh, cat_spec_wh, reid_dim, norm_wh, wh_weight, off_weight, id_weight, num_epochs, hm_weight, down_ratio, max_objs, track_buffer, image_mean, image_std, frame_rate, min_box_area)
```

Constructor parameters:
- **lr**: *float, default=0.1*  
  Specifies the initial learning rate to be used during training.
- **iters**: *int, default=-1*  
  Specifies the number if iteration per each training epoch. -1 means full epoch.
- **batch_size**: *int, default=4*  
  Specifies the size of the training batch.
- **optimizer**: *str {'sgd'}, default=sgd*  
  Specifies the optimizer type that should be used.
- **checkpoint_after_iter**: *int, default=0*  
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*  
  Specifies a path where the algorithm saves the onnx optimized model and checkpoints (if needed).
- **device**: *{'cpu', 'cuda', 'cuda:x'}, default='cuda'*  
  Specifies the device to be used.
- **max_dist**: *float, default=0.2*  
  Specifies the max cosine distance between re-id features.
- **min_confidence**: *float, default=0.3*  
  Specifies the minimal detection confidence to consider for tracking.
- **nms_max_overlap**: *float, default=0.5*  
  Specifies the max overlap value for non-max-suppression. Boxes that overlap more than this values are suppressed.
- **max_iou_distance**: *float, default=0.7*  
  Specifies the max IoU distance for detection-tracker matching.
- **max_age**: *int, default=70*  
  Specifies the max tracker age.
- **n_init**: *int, default=3*  
  Specifies the number of consecutive detections before the track is confirmed.
- **nn_budget**: *int, default=100*  
  Specifies the max samples per class for the nearest neighbor distance metric.


#### `ObjectTracking2DDeepSortLearner.fit`
```python
ObjectTracking2DDeepSortLearner.fit(
self, dataset, val_dataset, val_epochs, logging_path, silent, verbose, train_split_paths, val_split_paths, resume_optimizer, nID)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:
  - **dataset**: *object*  
    Object that holds the training dataset.
    Can be of type `ExternalDataset` (with type="market1501") or a custom dataset inheriting from `DatasetIterator`.
  - **epochs**: *int*  
    Specifies the number of epochs to train the model
  - **val_dataset**: *object, default=None*  
    Object that holds the validation dataset. If None, and the dataset is an `ExternalDataset`, dataset will be used to sample evaluation inputs. Can be of type `ExternalDataset` (with type="mot") or a custom dataset inheriting from `DatasetIterator`.
  - **val_epochs**: *int, default=-1*  
    Defines the number of train epochs passed to start evaluation. -1 means no evaluation. 
  - **logging_path**: *str, default=None*  
    Path to save log files. If set to None, only the console will be used for logging.
  - **silent**: *bool, default=False*  
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=False*  
    If set to True, enables maximum verbosity.
  - **train_transforms**: *object, default=None*  
    Specifies the `torchvision` transforms that should be applied to training data. If `None`, default transforms will apply.
  - **val_transforms**: *object, default=None*  
    Specifies the `torchvision` transforms that should be applied to validation data. If `None`, default transforms will apply.

#### `ObjectTracking2DDeepSortLearner.eval`
```python
ObjectTracking2DDeepSortLearner.eval(self, dataset, val_split_paths, logging_path, silent, verbose)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  

Parameters:
- **dataset**: *object*  
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` (with type="mot") or a custom dataset inheriting from `DatasetIterator`.
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=False*  
  If set to True, enables the maximum verbosity.

#### `ObjectTracking2DDeepSortLearner.infer`
```python
ObjectTracking2DDeepSortLearner.infer(self, batch, frame_ids)
```

This method is used to 2d object tracking on an image.
Returns a list of [TrackingAnnotationList](/src/opendr/engine/target.py#L545) objects if the list of [ImageWithDetections](/src/opendr/engine/data.py#L358) is given or a single [TrackingAnnotationList](/src/opendr/engine/target.py#L545) if a single [ImageWithDetections](/src/opendr/engine/data.py#L358) is given.

Parameters:
- **batch**: *engine.data.ImageWithDetections* or a *list of engine.data.ImageWithDetections*  
  Input data.
- **frame_ids**: *list of int, default=None*  
  Specifies frame ids for each input image to associate output tracking boxes. If None, -1 is used for each output box.

#### `ObjectTracking2DDeepSortLearner.save`
```python
ObjectTracking2DDeepSortLearner.save(self, path, verbose)
```

This method is used to save a trained model.
Provided with the path "/my/path/name" (absolute or relative), it creates the "name" directory, if it does not already exist.
Inside this folder, the model is saved as "name.pth" or "name.onnx" and the metadata file as "name.json".
If the directory already exists, the files are overwritten.

If [`self.optimize`](/src/opendr/perception/object_tracking_2d/fair_mot/object_tracking_2d_fair_mot_learner.py#L425) was run previously, it saves the optimized ONNX model in a similar fashion with an ".onnx" extension, by copying it from the `self.temp_path` it was saved previously during conversion.

Parameters:
- **path**: *str*  
  Path to save the model, including the filename.
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.

#### `ObjectTracking2DDeepSortLearner.load`
```python
ObjectTracking2DDeepSortLearner.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:
- **path**: *str*  
  Path of the model to be loaded.
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.

#### `ObjectTracking2DDeepSortLearner.optimize`
```python
ObjectTracking2DDeepSortLearner.optimize(self, do_constant_folding, img_size)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*  
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the operations that have all constant inputs, with pre-computed constant nodes.
- **img_size**: *(int, int), default=(64, 128)*  
  Specifies the size of an input image.

#### `ObjectTracking2DDeepSortLearner.download`
```python
@staticmethod
ObjectTracking2DDeepSortLearner.download(model_name, path, server_url)
```

Download utility for pretrained models.

Parameters:
- **model_name**: *str {'deep_sort'}*  
  The name of the model to download.
- **path**: *str*  
  Local path to save the downloaded files.
- **server_url**: *str, default=None*  
  URL of the pretrained models directory on an FTP server. If None, OpenDR FTP URL is used.


#### Examples

* **Training example using an `ExternalDataset`**.  
  Nano Market1501 dataset can be downloaded from the OpenDR server.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  import os
  import torch
  from opendr.perception.object_tracking_2d.deep_sort.object_tracking_2d_deep_sort_learner import (
      ObjectTracking2DDeepSortLearner,
  )
  from opendr.perception.object_tracking_2d.datasets.market1501_dataset import (
      Market1501Dataset,
  )

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "deep_sort"
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, name)

  dataset =  Market1501Dataset.download_nano_market1501(
      os.path.join(temp_dir, "market1501_dataset"), True
  )

  learner = ObjectTracking2DDeepSortLearner(
      temp_path=temp_dir,
      device=DEVICE,
  )

  learner.fit(
      dataset,
      epochs=2,
      val_epochs=2,
      verbose=True,
  )

  learner.save(model_path)
  ```

* **Training example using a `DatasetIterator`**.  
  If the `DatasetIterator` is given as a dataset, `val_dataset` should be specified.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  import os
  import torch
  from opendr.perception.object_tracking_2d.deep_sort.object_tracking_2d_deep_sort_learner import (
      ObjectTracking2DDeepSortLearner,
  )
  from opendr.perception.object_tracking_2d.datasets.market1501_dataset import (
      Market1501Dataset,
  )

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "deep_sort"
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, name)

  market1501_dataset_path =  Market1501Dataset.download_nano_market1501(
      os.path.join(temp_dir, "market1501_dataset"), True
  ).path

  dataset = Market1501DatasetIterator(
      os.path.join(market1501_dataset_path, "bounding_box_train"),
  )
  eval_dataset = Market1501DatasetIterator(
      os.path.join(market1501_dataset_path, "bounding_box_test"),
  )

  learner = ObjectTracking2DDeepSortLearner(
      temp_path=temp_dir,
      device=DEVICE,
  )

  learner.fit(
      dataset,
      epochs=2,
      val_dataset=eval_dataset,
      val_epochs=2,
      verbose=True,
  )

  learner.save(model_path)
  ```

* **Inference example.**
  ```python
  import os
  import torch
  from opendr.perception.object_tracking_2d.deep_sort.object_tracking_2d_deep_sort_learner import (
      ObjectTracking2DDeepSortLearner,
  )
  from opendr.perception.object_tracking_2d.datasets.market1501_dataset import (
      Market1501Dataset,
  )

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "deep_sort"
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, name)
  train_split_paths = {
      "nano_mot20": os.path.join(
          ".", "src", "perception", "object_tracking_2d",
          "datasets", "splits", "nano_mot20.train"
      )
  }

  mot_dataset_path = MotDataset.download_nano_mot20(
      os.path.join(temp_dir, "mot_dataset"), True
  ).path

  dataset = RawMotWithDetectionsDatasetIterator(
      mot_dataset_path,
      train_split_paths
  )

  learner = ObjectTracking2DDeepSortLearner(
    iters=3,
    num_epochs=1,
    checkpoint_after_iter=3,
    temp_path=temp_dir,
    device=DEVICE,
  )
  learner.load(model_path, verbose=True)

  result = learner.infer([
      dataset[0][0],
      dataset[1][0],
  ])

  print(result)
  ```

* **Optimization example for a previously trained model.**
  Inference can be run with the trained model after running `self.optimize`.
  ```python
  import os
  import torch
  from opendr.perception.object_tracking_2d.deep_sort.object_tracking_2d_deep_sort_learner import (
      ObjectTracking2DDeepSortLearner,
  )
  from opendr.perception.object_tracking_2d.datasets.market1501_dataset import (
      Market1501Dataset,
  )

  DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
  name = "deep_sort"
  temp_dir = "temp"
  model_path = os.path.join(temp_dir, name)
  train_split_paths = {
      "nano_mot20": os.path.join(
          ".", "src", "perception", "object_tracking_2d",
          "datasets", "splits", "nano_mot20.train"
      )
  }

  mot_dataset_path = MotDataset.download_nano_mot20(
      os.path.join(temp_dir, "mot_dataset"), True
  ).path

  dataset = RawMotWithDetectionsDatasetIterator(
      mot_dataset_path,
      train_split_paths
  )

  learner = ObjectTracking2DDeepSortLearner(
    iters=3,
    num_epochs=1,
    checkpoint_after_iter=3,
    temp_path=temp_dir,
    device=DEVICE,
  )
  learner.load(model_path, verbose=True)
  learner.optimize()

  result = learner.infer([
      dataset[0][0],
      dataset[1][0],
  ])

  print(result)
  ```

#### Performance Evaluation

The tests were conducted on the following computational devices:
- Intel(R) Xeon(R) Gold 6230R CPU on server
- Nvidia Jetson TX2
- Nvidia Jetson Xavier AGX
- Nvidia RTX 2080 Ti GPU on server with Intel Xeon Gold processors

Inference time is measured as the time taken to transfer the input to the model (e.g., from CPU to GPU), run inference using the algorithm, and return results to CPU.
Inner FPS refers to the speed of the model when the data is ready.
We report FPS (single sample per inference) as the mean of 100 runs.

Full FPS Evaluation of DeepSORT and FairMOT on MOT20 dataset
| Model    | TX2 (FPS) | Xavier (FPS) | RTX 2080 Ti (FPS) |
| -------- | --------- | ------------ | ----------------- |
| DeepSORT | 2.71      | 6.36         | 16.07             |
| FairMOT  | 0.79      | 2.36         | 10.42             |

Inner FPS Evaluation (model only) of DeepSORT and FairMOT on MOT20 dataset.
| Model    | TX2 (FPS) | Xavier (FPS) | RTX 2080 Ti (FPS) |
| -------- | --------- | ------------ | ----------------- |
| DeepSORT | 2.71      | 6.36         | 16.07             |
| FairMOT  | 0.79      | 2.36         | 17.16             |

Energy (Joules) of DeepSORT and FairMOT on embedded devices.
| Model    | TX2 (Joules) | Xavier (Joules) |
| -------- | ------------ | --------------- |
| DeepSORT | 11.27        | 3.72            |
| FairMOT  | 41.24        | 12.85           |

#### References
<a name="#object-tracking-2d-1" href="https://arxiv.org/abs/1703.07402">[1]</a> Simple Online and Realtime Tracking with a Deep Association Metric,
[arXiv](https://arxiv.org/abs/1703.07402).  
<a name="#object-tracking-2d-2" href="https://github.com/ZQPei/deep_sort_pytorch">[2]</a> Github: [ZQPei/deep_sort_pytorch](
https://github.com/ZQPei/deep_sort_pytorch)
