## binary_high_resolution module

The *binary high resolution* module contains the *BinaryHighResolutionLearner* class, which inherit from the abstract class *Learner*.

With this module one can train a fast convolutional model for binary classification of a single class. 
The model outputs a segmentation mask based on the confidence of the detection. 
The model is trained on VOC2012 format datasets. A test dataset can be downloaded using the tool's `download` method.
To create a custom dataset easily, [`labelImg`](https://github.com/heartexlabs/labelImg) is a good choice.
Refer to the train example [below](#examples) and the corresponding [demo](../../projects/python/perception/binary_high_resolution/train_eval_demo.py).

### Class BinaryHighResolutionLearner
Bases: `engine.learners.Learner`

The *BinaryHighResolutionLearner* class is a wrapper of the Binary High Resolution model [[1]](#binary).
It is used to train binary high resolution models on RGB images and run inference.
The [BinaryHighResolutionLearner](/src/opendr/perception/binary_high_resolution/binary_high_resolution_learner.py) class has the following public methods:

#### `BinaryHighResolutionLearner` constructor
```python
BinaryHighResolutionLearner(self, lr, iters, batch_size, optimizer, temp_path, device, weight_decay, momentum, num_workers, architecture)
```

Constructor parameters:

  - **lr**: *float, default=1e-3*\
    Learning rate during optimization.
  - **iters**: *int, default=100*\
    Number of epochs to train for.
  - **batch_size**: *int, default=512*\
    Dataloader batch size. Defaults to 1.
  - **optimizer**: *str, default="adam"*\
    Name of optimizer to use ("sgd" ,"rmsprop", or "adam").
  - **temp_path**: *str, default=''*\
    Path in which to store temporary files.
  - **device**: *str, default="cpu"*\
    Name of computational device ("cpu" or "cuda").
  - **weight_decay**: *float, default=1e-4*\
    Weight decay used for optimization.
  - **momentum**: *float, default=0.9*\
    Momentum used for optimization.
  - **num_workers**: *int, default=4*\
    Number of workers in dataloader.
  - **architecture**: *str, default="VGG_720p"*\
    Architecture to use ("VGG_720p" or "VGG_1080p").


#### `BinaryHighResolutionLearner.fit`
```python
BinaryHighResolutionLearner.fit(self, dataset, silent, verbose)
```

This method is used for training the algorithm on a train dataset.

Parameters:
  - **dataset**: *Dataset*\
    Training dataset.
  - **silent**: *bool, default=False*\
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=True*\
    If set to True, enables the maximum logging verbosity.


#### `BinaryHighResolutionLearner.eval`
```python
BinaryHighResolutionLearner.eval(self, dataset, silent, verbose)
```
This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:
  - **dataset**: *Dataset*\
    Dataset on which to evaluate model.
  - **silent**: *bool, default=False*\
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=True*\
    If set to True, enables the maximum logging verbosity.


#### `BinaryHighResolutionLearner.infer`
```python
BinaryHighResolutionLearner.infer(self, img)
```

This method is used to perform segmentation on an image.
Returns a `engine.target.Heatmap` object.

Parameters:
  - **img**: *Image*\
    Image to predict a heatmap.


#### `BinaryHighResolutionLearner.download`
```python
BinaryHighResolutionLearner.download(self, path, verbose, url)
```

Download pretrained models and testing images to path.

Parameters:
- **path**: *str, default="./demo_dataset"*\
  Path to where the test data folder will be saved.
- **verbose**: *bool, default=False*\
  If True, enables maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server and the tool's specific directory.


#### `BinaryHighResolutionLearner.save`
```python
BinaryHighResolutionLearner.save(self, path, verbose)
```

This method is used to save a trained model.
Provided with the path, absolute or relative, including a *folder* name, it creates a directory with the name
of the *folder* provided and saves the model inside with a proper format and a .json file with metadata.
If self.optimize was ran previously, it saves the optimized ONNX model in a similar fashion, by copying it
from the self.temp_path it was saved previously during conversion.

Parameters:
- **path**: *str*\
  Directory in which to save model weights and metadata.
- **verbose**: *bool, default=False*\
  If set to True, enables the maximum logging verbosity.


#### `BinaryHighResolutionLearner.load`
```python
BinaryHighResolutionLearner.load(self, path, verbose)
```

Loads the model from inside the path provided, based on the metadata .json file included.

Parameters:
- **path**: *str*\
  Local path to save the files.
- **verbose**: *bool, default=False*\
  If set to True, enables maximum logging verbosity.

#### `BinaryHighResolutionLearner.optimize`
```python
BinaryHighResolutionLearner.optimize(self, do_constant_folding)
```

Optimize method converts the model to ONNX format and saves the
model in the parent directory defined by self.temp_path. The ONNX model is then loaded and can be used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export. Constant-folding optimization will replace some of the ops that have all constant inputs with pre-computed constant nodes.


#### Examples

* **Training and evaluation example on test_dataset.**
  ```python  
  from opendr.perception.binary_high_resolution import BinaryHighResolutionLearner, visualize
  from opendr.engine.datasets import ExternalDataset
  
  
  if __name__ == '__main__':
      learner = BinaryHighResolutionLearner(')
      # Download dataset
      learner.download(verbose=True)
      # Prepare the dataset loader
      dataset = ExternalDataset("./demo_dataset", "VOC2012")
  
      learner.fit(dataset)
      learner.save("test_model")
      # Visualize the results
      visualize(learner, "./demo_dataset/test_img.png")
      print("Evaluation results = ", learner.eval(dataset))
  
  ```

* **Inference example on a single test image using a pretrained model.**
  ```python
  from opendr.perception.binary_high_resolution import BinaryHighResolutionLearner, visualize

  if __name__ == '__main__':
    # This example assumes that you have already trained a model using train_eval.demo.py
    learner = BinaryHighResolutionLearner()
    learner.load("test_model")
    # Visualize the results
    visualize(learner, "./demo_dataset/test_img.png")
  ```
  
#### Platform Compatibility

The platform compatibility evaluation is also reported below:

| Platform  | Compatibility Evaluation |
| ----------------------------------------------|-------|
| x86 - Ubuntu 20.04 (bare installation - CPU)  | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (bare installation - GPU)  | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (pip installation)         | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (CPU docker)               | :heavy_check_mark:   |
| x86 - Ubuntu 20.04 (GPU docker)               | :heavy_check_mark:   |
| NVIDIA Jetson TX2                             | :heavy_check_mark:   |
| NVIDIA Jetson Xavier AGX                      | :heavy_check_mark:   |

#### References
<a name="binary" href="https://www.sciencedirect.com/science/article/pii/S0031320320302107?via%3Dihub">[1]</a> Tzelepi, Maria, and Anastasios Tefas. "Improving the performance of lightweight CNNs for binary classification using quadratic mutual information regularization." Pattern Recognition 106 (2020): 107407.,
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320320302107?via%3Dihub).
