## attention_neural_bag_of_feature_learner module

The *attention_neural_bag_of_feature* module contains the *AttentionNeuralBagOfFeatureLearner* class, which inherits from the abstract class *Learner*.
In addition, the module also contains the convenient function to download and construct the AF dataset [[2]](#af), which is an ECG dataset for heart anomaly detection. 

### Class AttentionNeuralBagOfFeatureLearner 
Bases: `engine.learners.Learner`

The *AttentionNeuralBagOfFeatureLearner* class provides the implementation of the Attention-based Neural Bag-of-Features network for the heart anomaly detection problem [[1]](#anbof).
It can also be used to train a time-series classifier. 


The [AttentionNeuralBagOfFeatureLearner](/src/opendr/perception/heart_anomaly_detection/attention_neural_bag_of_feature/attention_neural_bag_of_feature_learner.py) class has the following public methods:

#### `AttentionNeuralBagOfFeatureLearner` constructor
```python
AttentionNeuralBagOfFeatureLearner(self, in_channels, series_length, n_class, n_codeword, quantization_type, attention_type, lr_scheduler, optimizer, weight_decay, dropout, iters, batch_size, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, test_mode)
```

**Parameters**:  

- **in_channels**: *int*  
  Specifies the number of univariate series in the input.  
- **series_length**: *int*    
  Specifies length of the input series.   
- **n_class**: *int*   
  Specifies the number of target classes.    
- **n_codeword**: *int, default=256*   
  Specifies the number of codewords in the bag-of-feature layer.   
- **quantization_type**: *{"nbof", "tnbof"}, default="nbof"*  
  Specifies the type of quantization layer.  
  There are two types of quantization layer: the logistic neural bag-of-feature layer ("nbof") or the temporal logistic bag-of-feature layer ("tnbof").  
 - **attention_type**: *{"spatial", "temporal", "spatialsa", "temporalsa", "spatiotemporal"}, default="spatial"*  
  Specifies the type of attention mechanism.  
  There are two types of attention: the spatial attention mechanism that focuses on the different codewords ("spatial") or the temporal attention mechanism that focuses on different temporal instances ("temporal"). Additionaly, there are three self-attention based mecahnisms as described [here](https://arxiv.org/abs/2201.11092).   
- **lr_scheduler**: *callable, default=`opendr.perception.heart_anomaly_detection.attention_neural_bag_of_feature.attention_neural_bag_of_feature_learner.get_cosine_lr_scheduler(2e-4, 1e-5)`*     
  Specifies the function that computes the learning rate, given the total number of epochs `n_epoch` and the current epoch index `epoch_idx`.  
  That is, the optimizer uses this function to determine the learning rate at a given epoch index.  
  Calling `lr_scheduler(n_epoch, epoch_idx)` should return the corresponding learning rate that should be used for the given epoch index.   
  The default `lr_scheduler` implements a schedule that gradually reduces the learning rate from the initial learning rate (`2e-4`) to the final learning rate (`1e-5`) using cosine function.  
  In order to use the default cosine learning rate scheduler with different initial and final learning rates, the user can use the convenient method `get_cosine_lr_scheduler(initial_lr, final_lr)` from this module, i.e., `opendr.perception.heart_anomaly_detection.attention_neural_bag_of_feature.attention_neural_bag_of_feature_learner.get_cosine_lr_scheduler`. 
  In addition, the convenient method from the same module `get_multiplicative_lr_scheduler(initial_lr, drop_at, multiplication_factor)` allows the user to specify a learning rate schedule that starts with an initial learning rate (`initial_lr`) and reduces the learning rate at certain epochs (specified by the list `drop_at`), using the given `multiplicative_factor`.   
- **optimizer**: *{'adam', 'sgd'}, default='adam'*  
  Specifies the name of optimizer.  
  If 'sgd' is used, momentum is set to 0.9 and nesterov is set to True.   
- **weight_decay**: *float, default=0.0*      
  Specifies the weight decay coefficient.     
- **dropout**: *float, default=0.2*     
  Specifies the dropout rate.   
- **iters**: *int, default=300*      
  Specifies the number of epochs used to train the model.      
- **batch_size**: *int, default=32*     
  Specifies the size of minit-batches.    
- **checkpoint_after_iter**: *int, default=1*  
  Specifies the frequency to save checkpoints.  
  The default behavior saves checkpoint after every epoch.    
- **checkpoint_load_iter**: *int, default=0*      
  Specifies the epoch index to load checkpoint.   
  If `checkpoint_load_iter=0`, training is done from scratch.  
  If `checkpoint_load_iter=-1`, training is done from the last checkpoint.  
  If `checkpoint_load_iter=k` and `k < iters`, training is done from checkpoint index `k`.  
  Note that loading from checkpoint is only available if `temp_path` argument is specified.   
- **temp_path**: *str, default=''*    
  Specifies path to the temporary directory that will be used to save checkpoints.  
  If not empty, this can be used to resume training later from the a particular checkpoint.   
- **device**: *{'cuda', 'cpu'}, default='cpu'*     
  Specifies the computation device.   
- **test_mode**: *bool, default=False*     
  If **test_mode** is True, only a small number of mini-batches is used for each epoch.  
  This option enables rapid testing of the code when training on large datasets.   

#### `AttentionNeuralBagOfFeatureLearner.fit`
```python
AttentionNeuralBagOfFeatureLearner.fit(self, train_set, val_set, test_set, class_weight, logging_path, silent, verbose)
```

This method is used for training the model using the provided train set.
If validation set is provided, it is used to validate the best model weights during the optimization process. 
That is, the final model weight is the one that produces the best validation F1 during optimization. 
If validation set is not provided, the final model weight is the one that produces the best training F1 during optimization.   

Returns a dictionary containing a list of accuracy, precision, recall, f1 measures (dict key: `"train_cross_entropy"`, `"train_acc"`, `"train_precision"`, `"train_recall"`, `"train_f1"`, `"val_cross_entropy"`, `"val_acc"`, `"val_precision"`, `"val_recall"`, `"val_f1"`, `"test_cross_entropy"`, `"test_acc"`, `"test_precision"`, `"test_recall"`, `"test_f1"`) during the entire optimization process. 
Note that the last value in the provided lists do not necessarily correspond to the final model performance due to the model selection policy mentioned above. 
To get the final performance on a dataset, please use the `eval` method of `AttentionNeuralBagOfFeatureLearner`.   
 
**Parameters**: 
 
  - **train_set**: *engine.datasets.DatasetIterator*   
    Object that holds the training set.  
    OpenDR dataset object, with `__getitem__` producing a pair of (`engine.data.Timeseries`, `engine.target.Category`).    
  - **val_set**: *engine.datasets.DatasetIterator, default=None*     
    Object that holds the validation set.  
    OpenDR dataset object, with `__getitem__` producing a pair of (`engine.data.Timeseries`, `engine.target.Category`).    
    If **val_set** is not `None`, it is used to select the model's weights that produce the best validation accuracy.   
  - **test_set**: *engine.datasets.DatasetIterator, default=None*    
    Object that holds the test set.  
    OpenDR dataset object, with `__getitem__` producing a pair of (`engine.data.Timeseries`, `engine.target.Category`).  
  - **class_weight**: *list, default=None*    
    A list that holds the class weight that can be used to counterbalance the effect of imbalanced dataset.   
  - **logging_path**: *str, default=''*    
    Tensorboard path. 
    If not empty, tensorboard data is saved to this path.    
  - **silent**: *bool, default=False*     
    If set to True, disables all printing, otherwise, the performance statistics, estimated time till finish are printed to STDOUT after every epoch.    
  - **verbose**: *bool, default=True*    
    If set to True, enables the progress bar of each epoch.   
 
**Returns**:

  - **performance**: *dict*   
    A dictionary that holds the performance curves with the following keys, with corresponding prefixes (`"train_"`, `"val_"`, `"test_"`):  
        - `"cross_entropy"`: a *list* that contains cross entropy measured after each training epoch.     
        - `"acc"`: a *list* that contains accuracy measured after each training epoch.   
        - `"precision"`: a *list* that contains precision measured after each training epoch.    
        - `"recall"`: a *list* that contains recall measured after each training epoch.   
        - `"f1"`: a *list* that contains f1 measured after each training epoch.     


#### `AttentionNeuralBagOfFeatureLearner.eval`
```python
AttentionNeuralBagOfFeatureLearner.eval(self, dataset, silent, verbose)
```

This method is used to evaluate the current model given the dataset.  
Returns a dictionary containing `"cross_entropy"`, `"acc"`, `"precision"`, `"recall"` and `"f1"` as keys.  
 
**Parameters**:

- **dataset**: *engine.datasets.DatasetIterator*   
  Object that holds the training set.  
  OpenDR dataset object, with `__getitem__` producing a pair of (`engine.data.Timeseries`, `engine.target.Category`).     
- **silent**: *bool, default=False*   
  If set to False, print the cross entropy and accuracy to STDOUT.    
- **verbose**: *bool, default=True*   
  If set to True, display a progress bar of the evaluation process.    
 
**Returns**:

- **performance**: *dict*    
  Dictionary that contains `"cross_entropy"`, `"acc"`, `"precision"`, `"recall"` and `"f1"`.   


#### `AttentionNeuralBagOfFeatureLearner.infer`
```python
AttentionNeuralBagOfFeatureLearner.infer(self, series)   
```

This method is used to generate the class prediction given an input series.    
Returns an instance of `engine.target.Category` representing the prediction.    

**Parameters**:

- **series**: *engine.data.Timeseries*   
  Object of type `engine.data.Timeseries` that holds the input data.    
 
**Returns**:

- **prediction**: *engine.target.Category*  
  Object of type `engine.target.Category` that contains the prediction.  


#### `AttentionNeuralBagOfFeatureLearner.save`
```python
AttentionNeuralBagOfFeatureLearner.save(self, path, verbose)
```

This method is used to save the current model instance under a given path. 
The saved model can be loaded later by calling `AttentionNeuralBagOfFeatureLearner.load(path)`  
Two files are saved under the given directory path, namely `metadata.json` and `model_weights.pt`. 
The former keeps the metadata and the latter keeps the model weights.

**Parameters**:

- **path**: *str*    
  Directory path to save the model.   
- **verbose**: *bool, default=True*    
  If set to True, print acknowledge message when saving is successful.    


#### `AttentionNeuralBagOfFeatureLearner.load`
```python
AttentionNeuralBagOfFeatureLearner.load(self, path, verbose)
```

This method is used to load a previously saved model (by calling `AttentionNeuralBagOfFeatureLearner.save(path)`) from a given directory. 
Note that under the given directory path, `metadata.json` and `model_weights.pt` must exist.   

**Parameters**: 
 
- **path**: *str*   
  Directory path of the model to be loaded.  
- **verbose**: *bool, default=True*     
  If set to True, print acknowledge message when model loading is successful.    

#### `AttentionNeuralBagOfFeatureLearner.download`
```python
AttentionNeuralBagOfFeatureLearner.download(self, path, fold_idx)
```

This method is used to download pretrained models for the AF dataset given different cross validation fold index. 
Pretrained models are available for `series_length=30` and `n_class=4`, `n_codeword in [256, 512]`, `quantization_type='nbof'` and `attention_type='temporal'`. 
The input series must be a univariate series, i.e., `in_channels=1`.   

**Parameters**:

- **path**: *str*   
  Directory path to download the model. Note that under this path, `metadata.json` and `model_weights.pt` will be downloaded, thus, to download different models, different paths should be given to avoid overwriting previously downloaded model. 
  In addition, the downloaded pretrained model weights can be loaded by calling `AttentionNeuralBagOfFeatureLearner.load(path)` afterward.      
- **fold_idx**: *{0, 1, 2, 3, 4}*   
  The index of the cross validation fold. 
  The AF dataset was divided into 5 folds and we provide the pretrained models for different cross-validation folds.  


#### `attention_neural_bag_of_feature.attention_neural_bag_of_feature_learner.get_AF_dataset`
```python
opendr.perception.heart_anomaly_detection.attention_neural_bag_of_feature.attention_neural_bag_of_feature_learner.get_AF_dataset(data_file, fold_idx, sample_length, standardize)
```

This method is used to download the pre-processed AF dataset [[2]](#af)   

**Parameters**: 

- **data_file**: *str*   
  Path to the data file. 
  If the given data file does not exist, it will be downloaded from the OpenDR server.     
- **fold_idx**: *{0, 1, 2, 3, 4}*   
  The index of the cross validation fold. 
  The AF dataset was divided into 5 folds.   
- **sample_length**: *int, default=30*    
  The length of each sample (in seconds). 
  This value must be at least 30 seconds.  
  Note that this value is different from `series_length`, which is equal to `sample_length` multiplied by the sampling rate (3000 Hz).    
- **standardize**: *bool, default=True*    
  Whether to standardize the input series.   

**Returns**:  

- **train_set**: *engine.datasets.DatasetIterator*   
  The training set that is constructed from `engine.datasets.DatasetIterator` that is ready to be used with the `fit()` method.    
- **val_set**: *engine.datasets.DatasetIterator*   
  The validation set that is constructed from `engine.datasets.DatasetIterator` that is ready to be used with the `fit()` method.    
- **series_length**: *int*   
  The length of each series (the number of elements in each series).  
  This parameter is used in the constructor of `AttentionNeuralBagOfFeatureLearner`.    
- **class_weight**: *list*   
  The weight given to each class, which is inversely proportional to the number of samples in each class.     


### Examples

* **Training example using the AF dataset**.  
In this example, we will train a heart anomaly detector using the AF dataset [[2]](#af), which contains single-lead ECG recordings and the class labels that categorize the samples into 4 classes: normal rhythm, atrial fibrillation, alternative rhythm and noise.
We start by importing the learner and the function used to download and construct the dataset.
This dataset has been preprocessed according to [[1]](#anbof) and split into 5-fold cross validation protocol.  

  ```python
  from opendr.perception.heart_anomaly_detection.attention_neural_bag_of_feature.attention_neural_bag_of_feature_learner import \
    AttentionNeuralBagOfFeatureLearner, get_cosine_lr_scheduler, get_AF_dataset 
  ```

  In order to download and construct the dataset, we will use the convenient function `get_AF_dataset` that has been imported above:

  ```python
  data_file = 'AF.dat'
  fold_idx = 0
  sample_length = 30
  train_set, val_set, series_length, class_weight = get_AF_dataset(data_file, fold_idx, sample_length)
  ```

  In the above snippet, we specify the name of the data file, the index of the cross validation fold and the length of each sample (in seconds). Here, we don't specify any absolute path for the data file so it will be downloaded and saved under the name "AF.dat" in the current directory.   

  Then we proceed to construct the learner object that will train the attention neural bag-of-feature network with 256 codewords and the temporal attention mechanism for 200 epochs starting from the learning rate of `2e-4` and gradually dropping to `1e-5` using a cosine scheduler. For the quantization layer, we will use the basic logistic version of the neural bag-of-features layer:  

  ```python
  learner = AttentionNeuralBagOfFeatureLearner(in_channels=1,
                                               series_length=series_length,
                                               n_class=4,
                                               n_codeword=256,
                                               quantization_type='nbof',
                                               attention_type='temporal',
                                               iters=200,
                                               lr_scheduler=get_cosine_lr_scheduler(2e-4, 1e-5))
  ```

  After the learner has been constructed, we can train the learner using the `fit` function:
   
  ```python
  performance = learner.fit(train_set, val_set, class_weight=class_weight) 
  ```

  In the above code, we pass the `class_weight` parameter to the `fit` function because the AF dataset is imbalanced. The `fit` function returns `performance`, which is a dictionary that contains the performance measured after each training epoch.   

* **Download and evaluate pretrained model for the AF dataset**.  
  In this example, we will show how pretrained models for the AF dataset [[2]](#af) can be easily downloaded by a single line of code using the functionality provided in `AttentionNeuralBagOfFeatureLearner` and evaluate the model on the test set.
  Since the AF dataset is divided into 5 folds, the following code iterates through each data split, downloads the corresponding pretrained model and evaluates on the corresponding validation set.  

  ```python
  from opendr.perception.heart_anomaly_detection.attention_neural_bag_of_feature.attention_neural_bag_of_feature_learner import \
    AttentionNeuralBagOfFeatureLearner, get_AF_dataset 
  import os

  # data file path
  data_file = 'AF.dat'

  # pretrained models directory
  pretrained_model_dir = 'models'
  if not os.path.exists(pretrained_model_dir):
      os.mkdir(pretrained_model_dir)

  # sample length of 30 seconds
  sample_length = 30

  # iterate through each validation fold
  # download and evaluate pretrained model
  for fold_idx in range(5):
      train_set, val_set, series_length, class_weight = get_AF_dataset(data_file, fold_idx, sample_length)
      learner = AttentionNeuralBagOfFeatureLearner(in_channels=1,
                                                   series_length=series_length,
                                                   n_class=4,
                                                   n_codeword=512,
                                                   quantization_type='nbof',
                                                   attention_type='temporal')

      # create a sub directory for each pretrained model
      model_path = os.path.join(pretrained_model_dir, 'AF_{}'.format(fold_idx))
      learner.download(model_path, fold_idx)

      # after pretrained model is downloaded to the given path
      # pretrained weights can be loaded with `load()`
      learner.load(model_path)

      # evaluate
      learner.eval(val_set)
  ```


#### References
<a name="anbof" href="https://arxiv.org/abs/2005.12250">[1]</a> Attention-based Neural Bag-of-Feautures Learning for Sequence Data,
[arXiv](https://arxiv.org/abs/2005.12250).  
<a name="af" href="https://physionet.org/content/challenge-2017/1.0.0/">[2]</a> AF Classification From A Short Single Lead ECG Recording, The PhysioNet Computing in Cardiology Challenge 2017.
