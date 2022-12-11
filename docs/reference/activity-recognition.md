## activity_recognition module

The *activity_recognition* module contains the *X3DLearner* and *CoX3DLearner* classes, which inherit from the abstract class *Learner*.

You can find the classes and the corresponding IDs regarding activity recognition [here](https://github.com/opendr-eu/opendr/blob/master/src/opendr/perception/activity_recognition/datasets/kinetics400_classes.csv).

### Class X3DLearner
Bases: `engine.learners.Learner`

The *X3DLearner* class is a wrapper of the X3D implementation found in the [SlowFast repository](https://github.com/facebookresearch/SlowFast) [[1]](#x3d).
It is used to train Human Activity Recognition models on RGB video clips and run inference.

X3D is a family of efficient models for video recognition, attaining state-of-the-art performance in offline recognition at multiple accuracy/efficiency trade-offs.

Pretrained X3D models are available [here](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md).
On [Kinetics-400](https://deepmind.com/research/open-source/kinetics), they achieve the following 1-clip accuracy:

| Model  | Accuracy |
| :---:  | :------: |
| X3D-XS | 54.68    |
| X3D-S  | 60.88    |
| X3D-M  | 63.84    |
| X3D-L  | 65.93    |


The [X3DLearner](/src/opendr/perception/activity_recognition/x3d/x3d_learner.py) class has the
following public methods:

#### `X3DLearner` constructor
```python
X3DLearner(self, lr, iters, batch_size, optimizer, lr_schedule, backbone, network_head, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, loss, weight_decay, momentum, drop_last, pin_memory, num_workers, seed, num_classes)
```

Constructor parameters:
  - **lr**: *float, default=1e-3*
    Learning rate during optimization.
  - **iters**: *int, default=10*
    Number of epochs to train for.
  - **batch_size**: *int, default=64*
    Dataloader batch size. Defaults to 64.
  - **optimizer**: *str, default="adam"*
    Name of optimizer to use ("sgd" or "adam").
  - **lr_schedule**: *str, default=""*
    Unused parameter.
  - **network_head**: *str, default="classification"*
    Head of network (only "classification" is currently available).
  - **checkpoint_after_iter**: *int, default=0*
    Unused parameter.
  - **checkpoint_load_iter**: *int, default=0*
    Unused parameter.
  - **temp_path**: *str, default=""*
    Path in which to store temporary files.
  - **device**: *str, default="cuda"*
    Name of computational device ("cpu" or "cuda").
  - **loss**: *str, default="cross_entropy"*
    Loss function used during optimization.
  - **weight_decay**: *[type], default=1e-5*
    Weight decay used for optimization.
  - **momentum**: *float, default=0.9*
    Momentum used for optimization.
  - **drop_last**: *bool, default=True*
    Drop last data point if a batch cannot be filled.
  - **pin_memory**: *bool, default=False*
    Pin memory in dataloader.
  - **num_workers**: *int, default=0*
    Number of workers in dataloader.
  - **seed**: *int, default=123*
    Random seed.
  - **num_classes**: *int, default=400*
    Number of classes to predict among.


#### `X3DLearner.fit`
```python
X3DLearner.fit(self, dataset, val_dataset, epochs, steps)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:
  - **dataset**: *Dataset*:
    Training dataset.
  - **val_dataset**: *Dataset, default=None*
    Validation dataset. If none is given, validation steps are skipped.
  - **epochs**: *int, default=None*
    Number of epochs. If none is supplied, self.iters will be used.
  - **steps**: *int, default=None*
    Number of training steps to conduct. If none, this is determined by epochs.


#### `X3DLearner.eval`
```python
X3DLearner.eval(self, dataset, steps)
```
This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.
Parameters:
  - **dataset**: *Dataset*
    Dataset on which to evaluate model.
  - **steps**: *int, default=None*
    Number of validation batches to evaluate. If None, all batches are evaluated.


#### `X3DLearner.infer`
```python
X3DLearner.infer(batch)
```

This method is used to perform classification of a video
Returns a list of `engine.target.Category` objects, where each holds a category.

Parameters:
- **batch**: *Union[engine.data.Video, List[engine.data.Video], torch.Tensor]*
  Video or batch of videos.
  The video should have shape (3, T, H, W). If a batch is supplied, its shape should be (B, 3, T, H, W).
  Here, B is the batch size, T is the clip length, and S is the spatial size in pixels.


#### `X3DLearner.save`
```python
X3DLearner.save(self, path)
```

Save model weights and metadata to path.
Provided with the path "/my/path/name" (absolute or relative), it creates the "name" directory, if it does not already
exist. Inside this folder, the model is saved as "model_name.pth" and the metadata file as "name.json".
If the files already exist, their names are versioned with a suffix.

If [`self.optimize`](/src/opendr/perception/activity_recognition/x3d/x3d_learner.py#L492) was run previously, it saves the optimized ONNX model in
a similar fashion with an ".onnx" extension.

Parameters:
- **path**: *str*
  Directory in which to save model weights and meta data.


#### `X3DLearner.load`
```python
X3DLearner.load(self, path)
```

This method is used to load a previously saved model from its saved folder.
Pretrained models from the [X3D model zoo](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md) can also be loaded using this function.

Parameters:
- **path**: *str*
  Path to metadata file in json format or to weights path.


#### `X3DLearner.optimize`
```python
X3DLearner.optimize(self, do_constant_folding)
```

Optimize model execution. This is acoomplished by saving to the ONNX format and loading the optimized model.

Parameters:
- **do_constant_folding**: *bool, default=False*
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the ops that have all constant inputs, with pre-computed constant nodes.


#### `X3DLearner.download`
```python
X3DLearner.download(self, path, model_names)
```

Download pretrained X3D models to path.

Parameters:
- **path**: *str*
  Local path to save the files.
- **model_names**: *set(str), default={"xs", "s", "m", "l"}*
  Names of the model sizes to download. Available model names are `{"xs", "s", "m", "l"}`


#### Examples

* **Fit model**.

  ```python
  from opendr.perception.activity_recognition import X3DLearner
  from opendr.perception.activity_recognition import KineticsDataset

  learner = X3DLearner(backbone="xs", device="cpu")
  train_ds = KineticsDataset(path="./datasets/kinetics400", frames_per_clip=4, split="train")
  val_ds = KineticsDataset(path="./datasets/kinetics400", frames_per_clip=4, split="val")
  learner.fit(dataset=train_ds, val_dataset=val_ds, logging_path="./logs")
  learner.save('./saved_models/trained_model')
  ```

* **Evaluate model**.

  ```python
  from opendr.perception.activity_recognition import X3DLearner
  from opendr.perception.activity_recognition import KineticsDataset

  learner = X3DLearner(backbone="xs", device="cpu")
  test_ds = KineticsDataset(path="./datasets/kinetics400", frames_per_clip=4, split="test")
  results = learner.eval(test_ds)  # Dict with accuracy and loss
  ```


* **Download pretrained model weights and initialize.**

  ```python
  from opendr.perception.activity_recognition import X3DLearner
  from pathlib import Path

  weights_path = Path("./weights/")
  X3DLearner.download(path=weights_path, model_names={"xs"})
  assert (weights_path / "x3d_xs.pyth").exists()
  learner = X3DLearner(backbone="xs", device="cpu").load(weights_path)
  ```


#### References
<a name="x3d" href="https://arxiv.org/abs/2004.04730">[1]</a> X3D: Expanding Architectures for Efficient Video Recognition,
[arXiv](https://arxiv.org/abs/2004.04730).


### Class CoX3DLearner
Bases: `engine.learners.Learner`

The *CoX3DLearner* class is a wrapper of CoX3D, the Continual version X3D.
It is used to train Human Activity Recognition models on RGB video clips and run inference frame-wise (one image at a time).

Continual networks introduce an alternative computational model, which lets a 3D-CNN (which otherwise must take a video-clip as input) compute outputs frame-by-frame.
This greatly speeds up inference in online-prediction, where predictions are computed for each new input-frame.
CoX3D is fully weight-compatible with pretrained X3D models.

Pretrained X3D models are available [here](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md).
On [Kinetics-400](https://deepmind.com/research/open-source/kinetics) using extended temporal window sizes, they achieve the following 1-clip accuracy when running in steady-state:


| Model     | Accuracy |
| :---:     | :------: |
| X3D-S_64  | 67.33    |
| X3D-M_64  | 71.03    |
| X3D-L_64  | 71.61    |


The [CoX3DLearner](/src/opendr/perception/activity_recognition/cox3d/cox3d_learner.py) class has the
following public methods:


#### `CoX3DLearner` constructor

```python
CoX3DLearner(self, lr, iters, batch_size, optimizer, lr_schedule, backbone, network_head, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, loss, weight_decay, momentum, drop_last, pin_memory, num_workers, seed, num_classes, temporal_window_size)
```

Constructor parameters:

  - **lr**: *float, default=1e-3*\
    Learning rate during optimization.
  - **iters**: *int, default=10*\
    Number of epochs to train for.
  - **batch_size**: *int, default=64*\
    Dataloader batch size. Defaults to 64.
  - **optimizer**: *str, default="adam"*\
    Name of optimizer to use ("sgd" or "adam").
  - **lr_schedule**: *str, default=""*\
    Unused parameter.
  - **network_head**: *str, default="classification"*\
    Head of network (only "classification" is currently available).
  - **checkpoint_after_iter**: *int, default=0*\
    Unused parameter.
  - **checkpoint_load_iter**: *int, default=0*\
    Unused parameter.
  - **temp_path**: *str, default=""*\
    Path in which to store temporary files.
  - **device**: *str, default="cuda"*\
    Name of computational device ("cpu" or "cuda").
  - **loss**: *str, default="cross_entropy"*\
    Loss function used during optimization.
  - **weight_decay**: *[type], default=1e-5*\
    Weight decay used for optimization.
  - **momentum**: *float, default=0.9*\
    Momentum used for optimization.
  - **drop_last**: *bool, default=True*\
    Drop last data point if a batch cannot be filled.
  - **pin_memory**: *bool, default=False*\
    Pin memory in dataloader.
  - **num_workers**: *int, default=0*\
    Number of workers in dataloader.
  - **seed**: *int, default=123*\
    Random seed.
  - **num_classes**: *int, default=400*\
    Number of classes to predict among.
  - **temporal_window_size**: *int, default=None*\
    Size of the final global average pooling.
    If None, size will be automatically chosen according to the backbone. Defaults to None.



#### `CoX3DLearner.fit`
Inherited from [X3DLearner](/src/opendr/perception/activity_recognition/x3d/x3d_learner.py)


#### `CoX3DLearner.eval`
Inherited from [X3DLearner](/src/opendr/perception/activity_recognition/x3d/x3d_learner.py)


#### `CoX3DLearner.infer`
```python
CoX3DLearner.infer(batch)
```
This method is used to perform classification of a video, image by image.
Returns a list of `engine.target.Category` objects, where each holds a category.

Parameters:

- **batch**: *Union[engine.data.Image, List[engine.data.Image], torch.Tensor]*\
  Image or batch of images.
  The image should have shape (3, H, W). If a batch is supplied, its shape should be (B, 3, H, W).
  Here, B is the batch size and S is the spatial size in pixels.


#### `CoX3DLearner.save`
Inherited from [X3DLearner](/src/opendr/perception/activity_recognition/x3d/x3d_learner.py)


#### `CoX3DLearner.load`
Inherited from [X3DLearner](/src/opendr/perception/activity_recognition/x3d/x3d_learner.py)


#### `CoX3DLearner.optimize`
Inherited from [X3DLearner](/src/opendr/perception/activity_recognition/x3d/x3d_learner.py)


#### `CoX3DLearner.download`
Inherited from [X3DLearner](/src/opendr/perception/activity_recognition/x3d/x3d_learner.py)


#### Examples

* **Fit model**.

  ```python
  from opendr.perception.activity_recognition import CoX3DLearner
  from opendr.perception.activity_recognition import KineticsDataset

  learner = CoX3DLearner(backbone="s", device="cpu")
  train_ds = KineticsDataset(path="./datasets/kinetics400", frames_per_clip=4, split="train")
  val_ds = KineticsDataset(path="./datasets/kinetics400", frames_per_clip=4, split="val")
  learner.fit(dataset=train_ds, val_dataset=val_ds, logging_path="./logs")
  learner.save('./saved_models/trained_model')
  ```

* **Evaluate model**.

  ```python
  from opendr.perception.activity_recognition import CoX3DLearner
  from opendr.perception.activity_recognition import KineticsDataset

  learner = CoX3DLearner(backbone="s", device="cpu")
  test_ds = KineticsDataset(path="./datasets/kinetics400", frames_per_clip=4, split="test")
  results = learner.eval(test_ds)  # Dict with accuracy and loss
  ```


* **Download pretrained model weights and initialize.**

  ```python
  from opendr.perception.activity_recognition import CoX3DLearner
  from pathlib import Path

  weights_path = Path("./weights/")
  CoX3DLearner.download(path=weights_path, model_names={"s"})
  assert (weights_path / "x3d_s.pyth").exists()
  learner = CoX3DLearner(backbone="s", device="cpu").load(weights_path)
  ```

* **Run frame-wise inference using extended temporal window size.**

  ```python
  from opendr.perception.activity_recognition import CoX3DLearner
  from pathlib import Path

  learner = CoX3DLearner(backbone="s", temporal_window_size=64).load(weights_path)

  # Prepare batch of images
  dl = torch.utils.data.DataLoader(
      KineticsDataset(path="./datasets/kinetics400", frames_per_clip=4, split="train"),
      batch_size=2,
      num_workers=0
  )
  video_batch = next(iter(dl))[0][:, :, 0]

  for i in range(video_batch.shape[2]):
      image_batch = batch[:, :, i]
      result = learner.infer(image_batch)
      ...
  ```


#### Performance Evaluation

TABLE-1: Input shapes, prediction accuracy on Kinetics 400, floating point operations (FLOPs), parameter count and maximum allocated memory of activity recognition learners at inference.
| Model   | Input shape (TxS2) | Acc. (%) | FLOPs (G) | Params (M) | Mem. (MB) |
| ------- | ------------------ | -------- | --------- | ---------- | --------- |
| X3D-L   | 16x3122            | 69.29    | 19.17     | 6.15       | 240.66    |
| X3D-M   | 16x2242            | 67.24    | 4.97      | 4.97       | 126.29    |
| X3D-S   | 13x1602            | 64.71    | 2.06      | 3.79       | 61.29     |
| X3D-XS  | 4x1602             | 59.37    | 0.64      | 3.79       | 28.79     |
| CoX3D-L | 1x3122             | 71.61    | 1.54      | 6.15       | 184.37    |
| CoX3D-M | 1x2242             | 71.03    | 0.40      | 4.97       | 68.96     |
| CoX3D-S | 1x1602             | 67.33    | 0.21      | 3.79       | 41.99     |


TABLE-2: Speed (evaluations/second) of activity recognition learner inference on various computational devices.
| Model   | CPU   | TX2  | Xavier | RTX 2080 Ti |
| ------- | ----- | ---- | ------ | ----------- |
| X3D-L   | 0.22  | 0.18 | 1.26   | 3.55        |
| X3D-M   | 0.75  | 0.69 | 4.50   | 6.94        |
| X3D-S   | 2.06  | 0.95 | 9.55   | 7.12        |
| X3D-XS  | 6.51  | 1.14 | 12.23  | 7.99        |
| CoX3D-L | 2.00  | 0.30 | 4.69   | 4.62        |
| CoX3D-M | 6.65  | 1.12 | 9.76   | 10.12       |
| CoX3D-S | 11.60 | 1.16 | 9.36   | 9.84        |


TABLE-3: Throughput (evaluations/second) of activity recognition learner inference on various computational devices.
The largest fitting power of two was used as batch size for each device.
| Model   | CPU   | TX2  | Xavier | RTX 2080 Ti |
| ------- | ----- | ---- | ------ | ----------- |
| X3D-L   | 0.22  | 0.21 | 1.73   | 3.55        |
| X3D-M   | 0.75  | 1.10 | 6.20   | 11.22       |
| X3D-S   | 2.06  | 2.47 | 7.83   | 29.51       |
| X3D-XS  | 6.51  | 6.50 | 38.27  | 78.75       |
| CoX3D-L | 2.00  | 0.62 | 10.40  | 14.47       |
| CoX3D-M | 6.65  | 4.32 | 44.07  | 105.64      |
| CoX3D-S | 11.60 | 8.22 | 64.91  | 196.54      |


TABLE-4: Energy (Joules) of activity recognition learner inference on embedded devices.
| Model   | TX2    | Xavier |
| ------- | ------ | ------ |
| X3D-L   | 187.89 | 23.54  |
| X3D-M   | 56.50  | 5.49   |
| X3D-S   | 33.58  | 2.00   |
| X3D-XS  | 26.15  | 1.45   |
| CoX3D-L | 117.34 | 5.27   |
| CoX3D-M | 24.53  | 1.74   |
| CoX3D-S | 22.79  | 2.07   |


TABLE-5: Human Activity Recognition platform compatibility evaluation.
| Platform                                     | Test results |
| -------------------------------------------- | ------------ |
| x86 - Ubuntu 20.04 (bare installation - CPU) | Pass         |
| x86 - Ubuntu 20.04 (bare installation - GPU) | Pass         |
| x86 - Ubuntu 20.04 (pip installation)        | Pass         |
| x86 - Ubuntu 20.04 (CPU docker)              | Pass         |
| x86 - Ubuntu 20.04 (GPU docker)              | Pass         |
| NVIDIA Jetson TX2                            | Pass\*       |
| NVIDIA Jetson Xavier AGX                     | Pass\*       |

*On NVIDIA Jetson devices, the Kinetics-400 dataset loader (dataset associated with available pretrained models) is not supported.
While import triggers an error in version 1.0 of the toolkit, a patch has been submitted, which avoids the import-error for the upcoming version.
Model inference works as expected.



#### References
<a name="x3d" href="https://arxiv.org/abs/2004.04730">[2]</a> X3D: Expanding Architectures for Efficient Video Recognition,
[arXiv](https://arxiv.org/abs/2004.04730).

