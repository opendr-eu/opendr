## multilinear_compressive_learning module

The *multilinear_compressive_learning* module contains the *MultilinearCompressiveLearner* class, which inherits from the abstract class *Learner*.

### Class MultilinearCompressiveLearner 
Bases: `opendr.engine.learners.Learner`

The *MultilinearCompressiveLearner* class provides the implementation of the Multilinear Compressive Learning Framework [[1]](#mcl).
The Multilinear Compressive Learning Framework optimizes the sensing operators of the compressive sensing device in conjuction with the learning model, which is usually deployed in a remote server.
The implementation is used to train and evaluate a multilinear compressive classification system for 3D tensor data, e.g. images.

The [MultilinearCompressiveLearner](#opendr.perception.compressive_learning.multilinear_compressive_learning.multilinear_compressive_learner.py) class has the following public methods:

#### `MultilinearCompressiveLearner` constructor
```python
MultilinearCompressiveLearner(self, input_shape, compressed_shape, backbone, n_class, pretrained_backbone, init_backbone, lr_scheduler, optimizer, weight_decay, n_init_iters, iters, batch_size, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, test_mode)
```

**Parameters**:

- **input_shape**: *tuple/list*  
  Specifies input shape of the data.
- **compressed_shape**: *tuple/list*  
  Specifies compressed shape of the compressed measurements.
- **backbone**: *str/torch.nn.Module*  
  Specifies the backbone classifier.  
  This can be a string that indicates a built-in backbone or an instance of `torch.nn.Module` that implements a custom backbone.  
  There are two types of built-in backbone: CIFAR backbones and ImageNet backbones. 
  The complete list of built-in backbones can be retrieved by calling `opendr.perception.compressive_learning.multilinear_compressive_learning.multilinear_compressive_learner.get_builtin_backbones()`, which includes:
	- 'cifar_allcnn'
	- 'cifar_vgg11'
	- 'cifar_vgg13'
	- 'cifar_vgg16'
	- 'cifar_vgg19'
	- 'cifar_resnet18'
	- 'cifar_resnet34'
	- 'cifar_resnet50'
	- 'cifar_resnet101'
	- 'cifar_resnet152'
	- 'cifar_densenet121'
	- 'cifar_densenet161'
	- 'cifar_densenet169'
	- 'cifar_densenet201'
	- 'imagenet_vgg11'
	- 'imagenet_vgg11_bn'
	- 'imagenet_vgg13'
	- 'imagenet_vgg13_bn'
	- 'imagenet_vgg16'
	- 'imagenet_vgg16_bn'
	- 'imagenet_vgg19'
	- 'imagenet_vgg19_bn'
	- 'imagenet_resnet18'
	- 'imagenet_resnet34'
	- 'imagenet_resnet50'
	- 'imagenet_resnet101'
	- 'imagenet_resnet152'
	- 'imagenet_resnext50_32x4d'
	- 'imagenet_resnext101_32x8d'
	- 'imagenet_wide_resnet50_2'
	- 'imagenet_wide_resnet101_2'
	- 'imagenet_densenet121'
	- 'imagenet_densenet161'
	- 'imagenet_densenet169'
	- 'imagenet_densenet201'

	For user-implemented backbones, the user can optionally implement `get_parameters()` method that returns two lists of parameters.  
	The first list should contain all parameters that will be optimized without **weight_decay** regularization and the second list should contain all parameters that will be optimized with the provided **weight_decay**.  
	All built-in backbones implement this by default. If `get_parameters()` is not implemented, the provided **weight_decay** is applied to all parameters of the user-implemented backbone during optimization.  
	
- **n_class**: *int*   
  Specifies the number of target classes. 
- **pretrained_backbone**: *{'', 'with_classifier', 'without_classifier'}, default=''*  
  Specifies whether to load pretrained weights for the built-in backbone.  
  If `pretrained_backbone='with_classifier'`, the weights of the last FC layer is also loaded.  
  In this case, `n_class` must be 10 or 100 for CIFAR backbones or 1000 for ImageNet backbones.  
  If `pretrained_backbone='without_classifier'`, the weights of the last FC layer is not loaded.  
  This allows loading intermediate layers of pretrained backbone. If `pretrained_backbone=''`, no pretrained weights are loaded.   
  When using a CIFAR pretrained backbone, the input images should be scaled to [0, 1], and then standardized with `mean = [0.491372, 0.482352, 0.446666]` and `std = [0.247058, 0.243529, 0.261568]`.  
  Similarly, when using the ImageNet pretrained backbone, the input images should be scaled to [0, 1] and then standardized with `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`.  
  
- **init_backbone**: *bool, default=True*  
  Specifies whether to initialize the backbone classifier by training it with uncompressed data.  
  This option can be used to skip the backbone pre-training step (by setting `init_backbone = False`) if the user-implemented backbone has been trained before, or pretrained built-in backbone is used.  
  
- **lr_scheduler**: *callable, default= `opendr.perception.compressive_learning.multilinear_compressive_learning.multilinear_compressive_learner.get_cosine_lr_scheduler(0.001, 0.00001)`*  
  Specifies the function that computes the learning rate, given the total number of epoch `n_epoch` and the current epoch index `epoch_idx`.  
  That is, the optimizer uses this function to determine the learning rate at a given epoch index.  
  Calling `lr_scheduler(n_epoch, epoch_idx)` should return the corresponding learning rate that should be used for the given epoch index.  
  The default `lr_scheduler` implements a schedule that gradually reduces the learning rate from the initial learning rate (0.001) to the final learning rate (0.00001) using cosine function.  
  In order to use the default cosine learning rate scheduler with different initial and final learning rates, the user can use the convenient method `get_cosine_lr_scheduler(initial_lr, final_lr)` from this module, i.e., `opendr.perception.compressive_learning.multilinear_compressive_learning.get_cosine_lr_scheduler`.  
  In addition, the convenient method from the same module `get_multiplicative_lr_scheduler(initial_lr, drop_at, multiplication_factor)` allows the user to specify a learning rate schedule that starts with an initial learning rate (`initial_lr`) and reduces the learning rate at certain epochs (specified by the list `drop_at`), using the given `multiplicative_factor`.  
  
- **optimizer**: *{'adam', 'sgd'}, default='adam'*   
  Specifies the name of optimizer.  
  If 'sgd' is used, momentum is set to 0.9 and nesterov is set to True.  
  
- **weight_decay**: *float, default=0.0001*   
  Specifies the weight decay coefficient.  
  Note that weight decay is not applied to batch norm parameters for built-in backbones by default.  
  The same behavior applied when user-implemented backbone has `get_parameters()` implemented as mentioned in **backbone** argument.  
  
- **n_init_iters**: *int, default=100*  
  Specifies the number of epochs used to initialize the backbone classifier and the sensing and feature synthesis components.  
  
- **iters**: *int, default=300*  
  Specifies the number of epochs used to train the all components in the multilinear compressive learning model.  
  
- **batch_size**: *int, default=32*   
  Specifies the size of minit-batches.  
  
- **checkpoint_after_iter**: *int, default=1*  
  Specifies the frequency to save checkpoints. 
  The default behavior saves checkpoint after every epoch.  
  
- **checkpoint_load_iter**: *{-1, 0}, default=0*   
  Specifies if training is done from scratch (`checkpoint_load_iter=0`) or training is done from the latest checkpoint (`checkpoint_load_iter=-1`). 
  Note that the latter option is only available if `temp_path` argument is specified.  
  
- **temp_path**: *str, default=''*  
  Specifies path to the temporary directory that will be used to save checkpoints.  
  If not empty, this can be used to resume training later from the latest checkpoint.  
  
- **device**: *{'cuda', 'cpu'}, default='cpu'*   
  Specifies the computation device.  
  
- **test_mode**: *bool, default=False*   
  If `test_mode` is True, only a small number of mini-batches is used for each epoch.  
  This option enables rapid testing of the code when training on large datasets.  
  

#### `MultilinearCompressiveLearner.fit`
```python
MultilinearCompressiveLearner.fit(self, train_set, val_set, test_set, logging_path, silent, verbose)
```

This method is used for training the multilinear compressive learning model using the provided train set. If validation set is provided, it is used to validate the best model weights during the optimization process. That is, the final model weight is the one that produces the best validation accuracy during optimization. If validation set is not provided, the final model weight is the one that produces the best training accuracy during optimization. 

Returns a dictionary containing a list of cross entropy measures (dict key: `"train_cross_entropy"`, `"val_cross_entropy"`, `"test_cross_entropy"`) and a list of accuracy (dict key: `"train_acc"`, `"val_acc"`, `"test_acc"`) during the entire optimization process. Note that the last value in the provided lists do not necessarily correspond to the final model performance due to the model selection policy mentioned above. To get the final performance on a dataset, please use the `eval` method of `MultilinearCompressiveLearner`.  
 
**Parameters**:

  - **train_set**: *engine.datasets.DatasetIterator*   
    Object that holds the training set.  
    OpenDR dataset object, with `__getitem__` producing a pair of (`engine.data.Image`, `engine.target.Category`).  
  - **val_set**: *engine.datasets.DatasetIterator, default=None*    
    Object that holds the validation set.  
    OpenDR dataset object, with `__getitem__` producing a pair of (`engine.data.Image`, `engine.target.Category`).  
    If **val_set** is not `None`, it is used to select the model's weights that produce the best validation accuracy.  
  - **test_set**: *engine.datasets.DatasetIterator, default=None*    
    Object that holds the test set.  
    OpenDR dataset object, with `__getitem__` producing a pair of (`engine.data.Image`, `engine.target.Category`).  
  - **logging_path**: *str, default=''*     
    Tensorboard path.  
    If not empty, tensorboard data is saved to this path.  
  - **silent**: *bool, default=False*     
    If set to True, disables all printing, otherwise, the performance statistics, estimated time till finish are printed to STDOUT after every epoch.  
  - **verbose**: *bool, default=True*   
    If set to True, enables the progress bar of each epoch.  
 
**Returns**:

  - **performance**: *dict*  
    A dictionary that holds the performance curves with the following keys:  
        - `backbone_performance`: a *dict* that contains `"train_cross_entropy"`,  `"val_cross_entropy"`, `"test_cross_entropy"`  when training the backbone classifier.  
        - `initialization_performance`: a *dict* that contains `"train_mean_squared_error"`, `"val_mean_squared_error"`, `"test_mean_squared_error"` when training the teacher's sensing and synthesis components.   
        - `compressive_learning_performance`: a *dict* that contains `"train_cross_entropy"`, `"train_acc"`, `"val_cross_entropy"`, `"val_acc"`, `"test_cross_entropy"`, `"test_acc"` when training the compressive model.   


#### `MultilinearCompressiveLearner.eval`   
```python
MultilinearCompressiveLearner.eval(self, dataset, silent, verbose)
```

This method is used to evaluate the current compressive learning model given the dataset.  
 
**Parameters**:

- **dataset**: *engine.datasets.DatasetIterator*   
  Object that holds the training set.  
  OpenDR dataset object, with `__getitem__` producing a pair of (`engine.data.Image`, `engine.target.Category`).  
- **silent**: *bool, default=False*     
  If set to False, print the cross entropy and accuracy to STDOUT.  
- **verbose**: *bool, default=True*   
  If set to True, display a progress bar of the evaluation process. 
 
**Returns**:

- **performance**: *dict*  
  Dictionary that contains `"cross_entropy"` and `"acc"`.  


#### `MultilinearCompressiveLearner.infer`  
```python
MultilinearCompressiveLearner.infer(img)
```

This method is used to generate the class prediction given a sample.  
Returns an instance of `engine.target.Category` representing the prediction.  

**Parameters**:

- **img**: *engine.data.Image*  
  Object of type `engine.data.Image` that holds the input data.  
 
**Returns**:

- **prediction**: *engine.target.Category*  
  Object of type `engine.target.Category` that contains the prediction.  

#### `MultilinearCompressiveLearner.infer_from_compressed_measurement`  
```python
MultilinearCompressiveLearner.infer_from_compressed_measurement(img)
```

This method is used to generate the class prediction given a compressed measurement.  
This method is used during deployment when the model receives compressed measurement from the sensor.  
Returns an instance of `engine.target.Category` representing the prediction.  

**Parameters**:

- **img**: *engine.data.Image*  
  Object of type `engine.data.Image` that holds the compressed measurement.  
 
**Returns**:

- **prediction**: *engine.target.Category*  
  Object of type `engine.target.Category` that contains the prediction.  

#### `MultilinearCompressiveLearner.get_sensing_parameters()`  
```python
MultilinearCompressiveLearner.get_sensing_parameters()
```

This method is used to get the parameters of the sensing component, which is used to setup the sensing device.  
Returns a list of numpy arrays.  

**Returns**:

- **params**: *list*  
  A list of numpy arrays that contain the parameters corresponding to each compressed dimension.  


#### `MultilinearCompressiveLearner.save`  
```python
MultilinearCompressiveLearner.save(path, verbose)
```

This method is used to save the current model instance under a given path. The saved model can be loaded later by calling `MultilinearCompressiveLearner.load(path)`. 
Two files are saved under the given directory path, namely `"path/metadata.json"` and `"path/model_weights.pt"`. The former keeps the metadata and the latter keeps the model weights. 

**Parameters**:

- **path**: *str*    
  Directory path to save the model    
- **verbose**: *bool, default=True*   
  If set to True, print acknowledge message when saving is successful.  
  

#### `MultilinearCompressiveLearner.load`  
```python
MultilinearCompressiveLearner.load(path, verbose)  
```

This method is used to load a previously saved model (by calling `MultilinearCompressiveLearner.save(path)`) from a given directory. Note that under the given directory path, `"metadata.json"` and `"model_weights.pt"` must exist. 

**Parameters**:

- **path**: *str*  
  Directory path of the model to be loaded.  
- **verbose**: *bool*, default to True   
  If set to True, print acknowledge message when model loading is successful.  

#### `MultilinearCompressiveLearner.download`
```python
MultilinearCompressiveLearner.download(path)
```

This method is used to download CIFAR-10 and CIFAR-100 pretrained models for the `cifar_allcnn` architecture. Pretrained models are available for `backbone='cifar_allcnn` and `n_class in [10, 10]` and `compressed_shape in [(20, 19, 2), (28, 27, 1), (14, 11, 2), (18, 17, 1), (9, 6, 1), (6, 9, 1)]`. Here we should note that the input image to the pretrained model should be scaled to the range [0, 1] and standardized using  `mean = [0.491372, 0.482352, 0.446666]` and `std = [0.247058, 0.243529, 0.261568]`. 

**Parameters**:
  
- **path**: *str*   
  Directory path to download the model. 
  Note that under this path, `"metadata.json"` and `"model_weights.pt"` will be downloaded, thus, to download different model, different paths should be given to avoid overwriting previously downloaded model.  
  In addition, the downloaded pretrained model weights can be loaded by calling `MultilinearCompressiveLearner.load(path)` afterward. 


### Examples

* **Training example using CIFAR-10 dataset**.  
  In this example, we will train a multilinear compressive learner for the CIFAR-10 dataset using built-in `cifar_allcnn` as the backbone classifier. For CIFAR-10 dataset, the `input_shape = (32, 32, 3)`. Let's assume that the compressive sensing device produces a measurement of shape 20x19x2, so `compressed_shape` is set to `(20, 19, 2)`. We start by first importing the `MultilinearCompressiveLearner` and the convenient function `get_cosine_lr_scheduler` to construct cosine learning rate schedule. 

  ```python
  from opendr.perception.compressive_learning.multilinear_compressive_learning.multilinear_compressive_learner import \
    MultilinearCompressiveLearner, get_cosine_lr_scheduler 
  ```

  Then, construct the learner object that will perform the initialization step for 100 epochs and train the compressive learner for 300 epochs starting from learning rate of `1e-3` and gradually dropping to `1e-5` using a cosine scheduler.

  ```python
  learner = MultilinearCompressiveLearner(input_shape=(32, 32, 3),
                                          compressed_shape=(20, 19, 2),
                                          backbone='cifar_allcnn',
                                          n_class=10,
                                          pretrained_backbone='',
                                          init_backbone=True,
                                          lr_scheduler=get_cosine_lr_scheduler(1e-3, 1e-5),
                                          n_init_iters=100,
                                          iters=300)
  ```

  In the above example, `pretrained_backbone=''` indicates that the pretrained weights for cifar dataset are not loaded and `init_backbone=True` indicates that during the initialization stage the backbone classifier will be trained using uncompressed data.

  After that, we will construct the train and test set for CIFAR-10 by wrapping around `torchvision.datasets.CIFAR` and inherit from OpenDR dataset class `engine.datasets.DatasetIterator`. Note that the `MultilinearCompressiveLearner.fit` function works with OpenDR dataset object (`engine.datasets.DatasetIterator`), which implements `__getitem__` and `__len__`, with `__getitem__` returning a pair of `(engine.data.Image, engine.target.Category)`. Although in this example we take advantage of the existing input pipeline from PyTorch, the same approach can also be used to wrap any existing input pipeline, e.g. from TensorFlow or MXNet, under OpenDR dataset object. 

  ```python
  # imports
  from opendr.engine.data import Image
  from opendr.engine.target import Category 
  from opendr.engine.datasets import DatasetIterator
  import numpy as np

  # dataset definition
  class Dataset(DatasetIterator):
      def __init__(self, dataset):
          self.dataset = dataset
          
      def __len__(self):
          return len(self.dataset)
          
      def __getitem__(self, i):
          x, y = self.dataset.__getitem__(i)
          # transpose so that the shape is height x width x channel 
          x = x.transpose(1, 2)
          x = x.transpose(0, 2)
          x = Image(x.numpy(), np.float32)
          y = Category(y)
          return x, y

  ```

  Then we will take advantage of the dataset provided in `torchvision` and create the necessary dataset objects
  
  ```python
  # imports
  import torchvision.datasets as datasets
  import torchvision.transforms as transforms

  # create transform functions
  pixel_mean = [0.491372, 0.482352, 0.446666]
  pixel_std = [0.247058, 0.243529, 0.261568]

  train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(),
                                        transforms.Normalize(pixel_mean, pixel_std)])

  test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(pixel_mean, pixel_std)])

  train_set = datasets.CIFAR10(root='.', train=True, download=True, transform=train_transform) 
  train_set = Dataset(train_set)
  test_set = datasets.CIFAR10(root='.', train=False, download=True, transform=test_transform) 
  test_set = Dataset(test_set)

  ```

  Finally, the model can be trained

  ```python
  performance = learner.fit(train_set, test_set) 
  ```

* **Download and evaluate pretrained model for CIFAR-10/CIFAR-100 dataset**.  
  In this example, we will create a multilinear compressive learner for the CIFAR10 dataset then download a pretrained model from OpenDR server, load the pretrained model weights and evaluate on the test set. The complete list of `compressed_shape` that has a pretrained model can be accessed via the constant `OpenDR.perception.compressive_learning.multilinear_compressive_learning.multilinear_compressive_learner.PRETRAINED_COMPRESSED_SHAPE`. For this example, we will use `compressed_shape = (20, 19, 2)`. 

  ```python
  from opendr.perception.compressive_learning.multilinear_compressive_learning.multilinear_compressive_learner import \
    MultilinearCompressiveLearner 
  from opendr.engine.data import Image
  from opendr.engine.target import Category 
  from opendr.engine.datasets import DatasetIterator
  import torchvision.datasets as datasets
  import torchvision.transforms as transforms
  import numpy as np
  import tempfile


  learner = MultilinearCompressiveLearner(input_shape=(32, 32, 3),
                                          compressed_shape=(20, 19, 2),
                                          backbone='cifar_allcnn',
                                          n_class=10)

  # in this example, we will use a temporary directory to download the model
  path = tempfile.TemporaryDirectory()

  # download the pretrained model
  learner.download(path.name)

  # load the pretrained weights from the given path
  learner.load(path.name)

  # clean up the temporary directory
  path.cleanup()

  # create the test set object and evaluate
  class Dataset(DatasetIterator):
      def __init__(self, dataset):
          self.dataset = dataset
          
      def __len__(self):
          return len(self.dataset)
          
      def __getitem__(self, i):
          x, y = self.dataset.__getitem__(i)
          # transpose so that the shape is height x width x channel 
          x = x.transpose(1, 2)
          x = x.transpose(0, 2)
          x = Image(x.numpy(), np.float32)
          y = Category(y)
          return x, y

  pixel_mean = [0.491372, 0.482352, 0.446666]
  pixel_std = [0.247058, 0.243529, 0.261568]

  test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(pixel_mean, pixel_std)])

  test_set = datasets.CIFAR10(root='.', train=False, download=True, transform=test_transform) 
  test_set = Dataset(test_set)

  # now we evaluate the pretrained model performance
  learner.eval(test_set)

  ```

  
#### References
<a name="mcl" href="https://arxiv.org/abs/1905.07481">[1]</a> Multilinear Compressive Learning,
[arXiv](https://arxiv.org/abs/1905.07481).  
