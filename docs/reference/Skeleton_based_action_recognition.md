## skeleton_based_action_recognition module

The *skeleton_based_action_recognition* module contains the *SpatioTemporalGCNLearner* and *ProgressiveSpatioTemporalGCNLearner* classes, which inherits from the abstract class *Learner*.

#### Data preparation  
  Download the NTU-RGB+D skeleton data from [here](https://github.com/shahroudy/NTURGB-D) and the kinetics-skeleton dataset from [here](https://drive.google.com/drive/folders/1SPQ6FmFsjGg3f59uCWfdUWI-5HJM_YhZ). 
  Then run the following function to preprocess the NTU-RGB+D and Kinetics skeleton data for ST-GCN methods: 
  
  ```python
  from OpenDR.perception.skeleton_based_action_recognition.algorithm.datasets import ntu_gendata
  from OpenDR.perception.skeleton_based_action_recognition.algorithm.datasets import kinetics_gendata
  python ntu_gendata.py --data_path ./data/nturgbd_raw_skeletons --ignored_sample_path ./algorithm/datasets/ntu_samples_with_missing_skeletons.txt --out_folder ./data/preprocessed_nturgbd
  python kinetics_gendata.py --data_path ./data/kinetics_raw_skeletons --out_folder ./data/preprocessed_kinetics_skeletons
  ```
  You need to specify the path of the downloaded data as `--data_path` and the path of the processed data as `--out_folder`. 
  ntu_samples_with_missing_skeletons.txt provides the NTU-RGB+D sample indices which don't contain any skeleton. 
  You need to specify the path of this file with --ignored_sample_path.

### Class SpatioTemporalGCNLearner
Bases: `engine.learners.Learner`

The *SpatioTemporalGCNLearner* class is a wrapper of the ST-GCN [[1]](#1) and the proposed methods TA-GCN [[2]](#2) and ST-BLN [[3]](#3) for Skeleton-based Human 
Action Recognition.
This implementation of ST-GCN can be found in [OpenMMLAB toolbox](
https://github.com/open-mmlab/mmskeleton/tree/b4c076baa9e02e69b5876c49fa7c509866d902c7).
It can be used to perform the baseline method ST-GCN and the proposed methods TA-GCN [[2]](#2) and ST-BLN [[3]](#3) for skeleton-based action recognition. 
The TA-GCN and ST-BLN methods are proposed on top of ST-GCN and make it more efficient in terms of number of model parameters and floating point operations. 

The [SpatioTemporalGCNLearner](#src.perception.skeleton_based_action_recognition.stgcn_learner.py) class has the
following public methods:

#### `SpatioTemporalGCNLearner` constructor
```python
SpatioTemporalGCNLearner(self, lr, batch_size, optimizer_name, lr_schedule,
                         checkpoint_after_iter, checkpoint_load_iter, temp_path,
                         device, num_workers, epochs, experiment_name,
                         device_ind, val_batch_size, drop_after_epoch,
                         start_epoch, dataset_name,method_name, 
                         stbln_symmetric, num_frames, num_subframes)
```

Constructor parameters:
- **lr**: *float, default=0.01*  
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=128*  
  Specifies number of skeleton sequences to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **optimizer_name**: *str {'sgd', 'adam'}, default='sgd'*  
  Specifies the optimizer type that should be used.
- **lr_schedule**: *str, default=' '*  
  Specifies the learning rate scheduler.
- **checkpoint_after_iter**: *int, default=0*
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*  
  Specifies a path where the algorithm saves the checkpoints and onnx optimized model (if needed).
- **device**: *{'cpu', 'cuda'}, default='cuda'*  
  Specifies the device to be used.
- **num_workers**: *int, default=32*  
  Specifies the number of workers to be used by the data loader.
- **epochs**: *int, default=50*  
  Specifies the number of epochs the training should run for.
- **experiment_name**: *str, default='stgcn_nturgbd'*  
  String name to attach to checkpoints.
- **device_ind**: *list, default=[0]*  
  List of GPU indices to be used if the device is 'cuda'. 
- **val_batch_size**: *int, default=256*  
  Specifies number of skeleton sequences to be bundled up in a batch during evaluation. This heavily affects memory usage, adjust according to your system.
- **drop_after_epoch**: *list, default=[30,40]*  
  List of epoch numbers in which the optimizer drops the learning rate.  
- **start_epoch**: *int, default=0*  
  Specifies the starting epoch number for training.  
- **dataset_name**: *str {'kinetics', 'nturgbd_cv', 'nturgbd_cs'}, default='nturgbd_cv'*  
  Specifies the name of dataset that is used for training and evaluation. 
- **method_name**: *str {'stgcn', 'stbln', 'tagcn'}, default='stgcn'*  
  Specifies the name of method to be trained and evaluated. For each method, a different model is trained. 
- **stbln_symmetric**: *bool, default=False*  
  Specifies if the random graph in stbln method is symmetric or not. This parameter is used if method_name is 'stbln'. 
- **num_frames**: *int, default=300*  
  Specifies the number of frames in each skeleton sequence. This parameter is used if the method_name is 'tagcn'. 
- **num_subframes**: *int, default=100*  
  Specifies the number of sub-frames that are going to be selected by the tagcn model. This parameter is used if the method_name is 'tagcn'.   


#### `SpatioTemporalGCNLearner.fit`
```python
SpatioTemporalGCNLearner.fit(self, dataset, val_dataset, logging_path, silent, verbose,
                             momentum, nesterov, weight_decay, train_data_filename,
                             train_labels_filename, val_data_filename,
                             val_labels_filename, skeleton_data_type)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Parameters:
  - **dataset**: *object*  
    Object that holds the training dataset.
    Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
  - **val_dataset**: *object*
    Object that holds the validation dataset. 
  - **logging_path**: *str, default=''*  
    Path to save TensorBoard log files and the training log files.
    If set to None or '', TensorBoard logging is disabled and no log file is created. 
  - **silent**: *bool, default=False*  
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=True***  
    If set to True, enables the maximum verbosity.
  - **momentum**: *float, default=0.9*  
    Specifies the momentum value for optimizer. 
  - **nesterov**: *bool, default=True***  
    If set to true, the optimizer uses nesterov.  
  - **weight_decay**: *float, default=0.0001***  
    Specifies the weight_decay value of the optimizer. 
  - **train_data_filename**: *str, default='train_joints.npy'*  
    Filename that contains the training data. 
    This file should be contained in the dataset path provided.
    Note that this is a file name, not a path.
  - **train_labels_filename**: *str, default='train_labels.pkl'*  
    Filename of the labels .pkl file. 
    This file should be contained in the dataset path provided.
  - **val_data_filename**: *str, default='val_joints.npy'*  
    Filename that contains the validation data.
    This file should be contained in the dataset path provided.
    Note that this is a filename, not a path.
  - **val_labels_filename**: *str, default='val_labels.pkl'*  
    Filename of the validation labels .pkl file.
    This file should be contained in the dataset path provided.
  - **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
    The data stream that should be used for training and evaluation. 
    
#### `SpatioTemporalGCNLearner.eval`
```python
SpatioTemporalGCNLearner.eval(self, val_dataset, epoch, silent, verbose,
                              val_data_filename, val_labels_filename, skeleton_data_type,
                              save_score, wrong_file, result_file, show_topk)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:
- **val_dataset**: *object*  
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **epoch**: *int, default=0*  
  The training epoch in which the model is evaluated. 
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*  
  If set to True, enables the maximum verbosity.
- **val_data_filename**: *str, default='val_joints.npy'*  
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **val_labels_filename**: *str, default='val_labels.pkl'*  
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.
- **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
  The data stream that should be used for training and evaluation. 
- **save_score**: *bool, default=False*  
  If set to True, it saves the classification score of all samples in differenc classes
  in a log file. Default to False.
- **wrong_file**: *str, default=None*  
  If set to True, it saves the results of wrongly classified samples. Default to False.
- **result_file**: *str, default=None*  
  If set to True, it saves the classification results of all samples. Default to False.
- **show_topk**: *list, default=[1, 5]*  
  Is set to a list of integer numbers defining the k in top-k accuracy. Default is set to [1,5].
  
#### `SpatioTemporalGCNLearner.init_model`
```python
SpatioTemporalGCNLearner.init_model(self)
```
This method is used to initialize the imported model and its loss function. 
  

#### `SpatioTemporalGCNLearner.infer`
```python
SpatioTemporalGCNLearner.infer(self, SkeletonSeq_batch)
```
This method is used to perform action recognition on a sequence of skeletons. 
It returns the action category as an object of `engine.target.Category` if a proper input object `engine.data.SkeletonSequence` is given. 

Parameters:
- **SkeletonSeq_batch**: *object***  
  Object of type engine.data.SkeletonSequence.

#### `SpatioTemporalGCNLearner.save`
```python
SpatioTemporalGCNLearner.save(self, path, model_name, verbose)
```
This method is used to save a trained model.
Provided with the path "/my/path" (absolute or relative), it creates the "path" directory, if it does not already 
exist. Inside this folder, the model is saved as "model_name.pt" and the metadata file as "model_name.json". If the directory
already exists, the "model_name.pt" and "model_name.json" files are overwritten.

If [`self.optimize`](#SpatioTemporalGCNLearner.optimize) was run previously, it saves the optimized ONNX model in 
a similar fashion with an ".onnx" extension, by copying it from the self.temp_path it was saved previously 
during conversion.

Parameters:
- **path**: *str*  
  Path to save the model.
- **model_name**: *str*  
  The file name to be saved. 
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.

#### `SpatioTemporalGCNLearner.load`
```python
SpatioTemporalGCNLearner.load(self, path, model_name, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:
- **path**: *str*  
  Path of the model to be loaded.
- **model_name**: *str*  
  The file name to be loaded. 
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.


#### `SpatioTemporalGCNLearner.optimize`
```python
SpatioTemporalGCNLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*  
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the operations that have all constant inputs, with pre-computed constant nodes.



#### `SpatioTemporalGCNLearner.multi_stream_eval`
```python
SpatioTemporalGCNLearner.multi_stream_eval(self, dataset, scores, data_filename,
                                           labels_filename, skeleton_data_type,
                                           verbose, silent)
```
This method is used to ensemble the classification results of the model on two or more data streams like joints, bones and motions. 
It returns the top-k classification performance of ensembled model. 

Parameters:
- **dataset**: *object*  
  Object that holds the dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **score**: *list*  
  A list of score arrays. Each array in the list contains the evaluation results for a data stream.
- **data_filename**: *str, default='val_joints.npy'*  
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **labels_filename**: *str, default='val_labels.pkl'*  
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.
- **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
  The data stream that should be used for training and evaluation. 
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*  
  If set to True, enables the maximum verbosity.


#### `SpatioTemporalGCNLearner.download`
```python
@staticmethod
SpatioTemporalGCNLearner.download(self, path, mode, verbose, url, file_name)
```

Download utility for various skeleton-based action recognition components. Downloads files depending on mode and
saves them in the path provided. It supports downloading:
1. the pretrained weights for stgcn, tagcn and stbln models. 
2. a dataset containing one or more skeleton sequences and its labels.  

Parameters:
- **path**: *str, default=None*  
  Local path to save the files, defaults to self.parent_dir if None.
- **mode**: *str, default="pretrained"*  
  What file to download, can be one of "pretrained", "train_data", "val_data", "test_data"
- **verbose**: *bool, default=False*  
  Whether to print messages in the console.
- **url**: *str, default=OpenDR FTP URL*  
  URL of the FTP server.
- **file_name**: *str*  
  The name of the file containing the pretrained model. 


#### Examples

* **Training example using an `ExternalDataset`**.  
  The training and evaluation dataset should be present in the path provided, along with the labels file.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  from OpenDR.perception.skeleton_based_action_recognition.stgcn_learner import SpatioTemporalGCNLearner
  from OpenDR.engine.datasets import ExternalDataset
  
  training_dataset = ExternalDataset(path='./data/preprocessed_nturgbd/xview', dataset_type='NTURGBD')
  validation_dataset = ExternalDataset(path='./data/preprocessed_nturgbd/xview', dataset_type='NTURGBD')
 
  stgcn_learner = SpatioTemporalGCNLearner(temp_path='./parent_dir',
                                            batch_size=64, epochs=50,
                                            checkpoint_after_iter=10, val_batch_size=128,
                                            dataset_name='nturgbd_cv',
                                            experiment_name='stgcn_nturgbd',
                                            method_name='stgcn')

   stgcn_learner.fit(dataset=training_dataset, val_dataset=validation_dataset, logging_path='./logs', silent=True,
                    train_data_filename='train_joints.npy',
                    train_labels_filename='train_labels.pkl', val_data_filename='val_joints.npy',
                    val_labels_filename='val_labels.pkl',
                    skeleton_data_type='joint')
  stgcn_learner.save(path='./saved_models/stgcn_nturgbd_cv_checkpoints', model_name='test_stgcn')
  ```
  In a similar manner train the TA-GCN model by specifying the number of important frames that the model selects as num_subframes. 
  The number of frames in both NTU-RGB+D and Kinetics-skeleton is 300.  
  
  ```python
  tagcn_learner = SpatioTemporalGCNLearner(temp_path='./parent_dir',
                                            batch_size=64, epochs=50,
                                            checkpoint_after_iter=10, val_batch_size=128,
                                            dataset_name='nturgbd_cv',
                                            experiment_name='tagcn_nturgbd',
                                            method_name='tagcn', num_frames=300, num_subframes=100)

  tagcn_learner.fit(dataset=training_dataset, val_dataset=validation_dataset, logging_path='./logs', silent=True,
                    train_data_filename='train_joints.npy',
                    train_labels_filename='train_labels.pkl', val_data_filename='val_joints.npy',
                    val_labels_filename='val_labels.pkl',
                    skeleton_data_type='joint')
  tagcn_learner.save(path='./saved_models/tagcn_nturgbd_cv_checkpoints', model_name='test_tagcn')
  ```
  
  For training the ST-BLN model, set the method_name to 'stbln' and specify if the model uses a symmetric attention matrix or not by setting stbln_symmetric to True or False. 
  
  ```python

  stbln_learner = SpatioTemporalGCNLearner(temp_path='./parent_dir',
                                            batch_size=64, epochs=50,
                                            checkpoint_after_iter=10, val_batch_size=128,
                                            dataset_name='nturgbd_cv',
                                            experiment_name='stbln_nturgbd',
                                            method_name='stbln', stbln_symmetric=False)

  stbln_learner.fit(dataset=training_dataset, val_dataset=validation_dataset, logging_path='./logs', silent=True,
                    train_data_filename='train_joints.npy',
                    train_labels_filename='train_labels.pkl', val_data_filename='val_joints.npy',
                    val_labels_filename='val_labels.pkl',
                    skeleton_data_type='joint')
  stbln_learner.save(path='./saved_models/stbln_nturgbd_cv_checkpoints', model_name='test_stbln')
  ```
  

* **Inference on a test skeleton sequence**
  ```python
  from OpenDR.perception.skeleton_based_action_recognition.stgcn_learner import SpatioTemporalGCNLearner
  import numpy
  stgcn_learner = SpatioTemporalGCNLearner(temp_path='./parent_dir',
                                            batch_size=64, epochs=50,
                                            checkpoint_after_iter=10, val_batch_size=128,
                                            dataset_name='nturgbd_cv',
                                            experiment_name='stgcn_nturgbd',
                                            method_name='stgcn')
  # Download the default pretrained stgcn model in the parent_dir
  stgcn_learner.download(
            mode="pretrained", path='./parent_dir/pretrained_models', file_name='pretrained_stgcn')  
  
  stgcn_learner.load('./parent_dir/pretrained_models', model_name='pretrained_stgcn')
  test_data_path = stgcn_learner.download(mode="test_data")  # Download a test data 
  test_data = numpy.load(test_data_path)
  action_category = stgcn_learner.infer(test_data)
  
  ```
  
* **Optimization example for a previously trained model.**
  Inference can be run with the trained model after running self.optimize.
  ```python
  from OpenDR.perception.skeleton_based_action_recognition.stgcn_learner import SpatioTemporalGCNLearner


  stgcn_learner = SpatioTemporalGCNLearner(temp_path='./parent_dir',
                                            batch_size=64, epochs=50,
                                            checkpoint_after_iter=10, val_batch_size=128,
                                            dataset_name='nturgbd_cv',
                                            experiment_name='stgcn_nturgbd',
                                            method_name='stgcn')
  stgcn_learner.download(
            mode="pretrained", path='./parent_dir/pretrained_models', file_name='pretrained_stgcn')  
  
  stgcn_learner.load(path='./parent_dir/pretrained_models', file_name='pretrained_stgcn')
  stgcn_learner.optimize(do_constant_folding=True)
  stgcn_learner.save(path='./parent_dir/optimized_model', model_name='optimized_stgcn')
  ```
  The inference and optimization can be performed for TA-GCN and ST-BLN methods in a similar manner only by specifying the method_name to 'tagcn' or 'stbln', respectively in the learner class constructor. 


### Class ProgressiveSpatioTemporalGCNLearner
Bases: `engine.learners.Learner`

The *ProgressiveSpatioTemporalGCNLearner* class is an implementation of the proposed method PST-GCN [[4]](#4) for Skeleton-based Human 
Action Recognition.
It finds an optimized and data dependant spatio-temporal graph convolutional network topology for skeleton-based action recognition. 
The [ProgressiveSpatioTemporalGCNLearner](#src.perception.skeleton_based_action_recognition.pstgcn_learner.py) class has the
following public methods:


#### `ProgressiveSpatioTemporalGCNLearner` constructor
```python
ProgressiveSpatioTemporalGCNLearner(self, lr, batch_size, optimizer_name, lr_schedule,
                                    checkpoint_after_iter, checkpoint_load_iter, temp_path,
                                    device, num_workers, epochs, experiment_name,
                                    device_ind, val_batch_size, drop_after_epoch,
                                    start_epoch, dataset_name, 
                                    blocksize, numblocks, numlayers, topology,
                                    layer_threshold, block_threshold)
```

Constructor parameters:
- **lr**: *float, default=0.01*  
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=128*  
  Specifies number of skeleton sequences to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **optimizer_name**: *str {'sgd', 'adam'}, default='sgd'*  
  Specifies the optimizer type that should be used.
- **lr_schedule**: *str, default=' '*  
  Specifies the learning rate scheduler.
- **checkpoint_after_iter**: *int, default=0*
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*  
  Specifies a path where the algorithm saves the checkpoints and onnx optimized model (if needed).
- **device**: *{'cpu', 'cuda'}, default='cuda'*  
  Specifies the device to be used.
- **num_workers**: *int, default=32*  
  Specifies the number of workers to be used by the data loader.
- **epochs**: *int, default=50*  
  Specifies the number of epochs the training should run for.
- **experiment_name**: *str, default='stgcn_nturgbd'*  
  String name to attach to checkpoints.
- **device_ind**: *list, default=[0]*  
  List of GPU indices to be used if the device is 'cuda'. 
- **val_batch_size**: *int, default=256*  
  Specifies number of skeleton sequences to be bundled up in a batch during evaluation. This heavily affects memory usage, adjust according to your system.
- **drop_after_epoch**: *list, default=[30,40]*  
  List of epoch numbers in which the optimizer drops the learning rate.  
- **start_epoch**: *int, default=0*  
  Specifies the starting epoch number for training.  
- **dataset_name**: *str {'kinetics', 'nturgbd_cv', 'nturgbd_cs'}, default='nturgbd_cv'*  
  Specifies the name of dataset that is used for training and evaluation. 
- **block_size**: *int, default=20*  
  Specifies the number of output channels (or neurons) that are added to each layer of the network at each progression iteration. 
- **numblocks**: *int, default=10*  
  Specifies the maximum number of blocks that are added to each layer of the network at each progression iteration. 
- **numlayers**: *int, default=10*  
  Specifies the maximum number of layers that are built for the network.
- **topology**: *list, default=[]*  
  Specifies the initial topology of the network. The default is set to [], since the method gets an empty network as input and builds it progressively. 
- **layer_threshold**: *float, default=1e-4*  
  Specifies the threshold which is used by the method to identify when it should stop adding new layers.  
- **block_threshold**: *float, default=1e-4*  
  Specifies the threshold which is used by the model to identify when it should stop adding new blocks in each layer.  
  

#### `ProgressiveSpatioTemporalGCNLearner.fit`
```python
ProgressiveSpatioTemporalGCNLearner.fit(self, dataset, val_dataset, logging_path, silent, verbose,
                                        momentum, nesterov, weight_decay, train_data_filename,
                                        train_labels_filename, val_data_filename,
                                        val_labels_filename, skeleton_data_type)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Parameters:
- **dataset**: *object*  
Object that holds the training dataset.
Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **val_dataset**: *object*
Object that holds the validation dataset. 
- **logging_path**: *str, default=''*  
Path to save TensorBoard log files and the training log files.
If set to None or '', TensorBoard logging is disabled and no log file is created. 
- **silent**: *bool, default=False*  
If set to True, disables all printing of training progress reports and other information to STDOUT.
- **verbose**: *bool, default=True***  
If set to True, enables the maximum verbosity.
- **momentum**: *float, default=0.9*  
Specifies the momentum value for optimizer. 
- **nesterov**: *bool, default=True***  
If set to true, the optimizer uses nesterov.  
- **weight_decay**: *float, default=0.0001***  
Specifies the weight_decay value of the optimizer. 
- **train_data_filename**: *str, default='train_joints.npy'*  
Filename that contains the training data. 
This file should be contained in the dataset path provided.
Note that this is a file name, not a path.
- **train_labels_filename**: *str, default='train_labels.pkl'*  
Filename of the labels .pkl file. 
This file should be contained in the dataset path provided.
- **val_data_filename**: *str, default='val_joints.npy'*  
Filename that contains the validation data.
This file should be contained in the dataset path provided.
Note that this is a filename, not a path.
- **val_labels_filename**: *str, default='val_labels.pkl'*  
Filename of the validation labels .pkl file.
This file should be contained in the dataset path provided.
- **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
The data stream that should be used for training and evaluation. 


#### `ProgressiveSpatioTemporalGCNLearner.eval`
```python
ProgressiveSpatioTemporalGCNLearner.eval(self, val_dataset, epoch, silent, verbose,
                                         val_data_filename, val_labels_filename, skeleton_data_type,
                                         save_score, wrong_file, result_file, show_topk)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:

- **val_dataset**: *object*  
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **epoch**: *int, default=0*  
  The training epoch in which the model is evaluated. 
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*  
  If set to True, enables the maximum verbosity.
- **val_data_filename**: *str, default='val_joints.npy'*  
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **val_labels_filename**: *str, default='val_labels.pkl'*  
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.
- **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
  The data stream that should be used for training and evaluation. 
- **save_score**: *bool, default=False*  
  If set to True, it saves the classification score of all samples in differenc classes
  in a log file. Default to False.
- **wrong_file**: *str, default=None*  
  If set to True, it saves the results of wrongly classified samples. Default to False.
- **result_file**: *str, default=None*  
  If set to True, it saves the classification results of all samples. Default to False.
- **show_topk**: *list, default=[1, 5]*  
  Is set to a list of integer numbers defining the k in top-k accuracy. Default is set to [1,5].


#### `ProgressiveSpatioTemporalGCNLearner.init_model`
```python
ProgressiveSpatioTemporalGCNLearner.init_model(self)
```
This method is used to initialize the imported model and its loss function. 
  
  
#### `ProgressiveSpatioTemporalGCNLearner.network_builder`
```python
ProgressiveSpatioTemporalGCNLearner.network_builder(self, dataset, val_dataset, train_data_filename,
                                                    train_labels_filename, val_data_filename,
                                                    val_labels_filename, skeleton_data_type, verbose)
```
This method implement the ST-GCN Augmentation Module (ST-GCN-AM) which builds the network topology progressively. 
Parameters:

- **dataset**: *object*  
  Object that holds the training dataset.
- **val_dataset**: *object*  
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **train_data_filename**: *str, default='train_joints.npy'*  
  Filename that contains the training data. 
  This file should be contained in the dataset path provided.
  Note that this is a file name, not a path.
- **train_labels_filename**: *str, default='train_labels.pkl'*  
  Filename of the labels .pkl file. 
  This file should be contained in the dataset path provided.
- **val_data_filename**: *str, default='val_joints.npy'*  
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **val_labels_filename**: *str, default='val_labels.pkl'*  
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.
- **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
  The data stream that should be used for training and evaluation. 
- **verbose**: *bool, default=True***  
  Whether to print messages in the console.
  
  
#### `ProgressiveSpatioTemporalGCNLearner.infer`
```python
ProgressiveSpatioTemporalGCNLearner.infer(self, SkeletonSeq_batch)
```
This method is used to perform action recognition on a sequence of skeletons. 
It returns the action category as an object of `engine.target.Category` if a proper input object `engine.data.SkeletonSequence` is given. 

Parameters:
- **SkeletonSeq_batch**: *object***  
  Object of type engine.data.SkeletonSequence.

#### `ProgressiveSpatioTemporalGCNLearner.save`
```python
ProgressiveSpatioTemporalGCNLearner.save(self, path, model_name, verbose)
```
This method is used to save a trained model.
Provided with the path "/my/path" (absolute or relative), it creates the "path" directory, if it does not already 
exist. Inside this folder, the model is saved as "model_name.pt" and the metadata file as "model_name.json". If the directory
already exists, the "model_name.pt" and "model_name.json" files are overwritten.

If [`self.optimize`](#SpatioTemporalGCNLearner.optimize) was run previously, it saves the optimized ONNX model in 
a similar fashion with an ".onnx" extension, by copying it from the self.temp_path it was saved previously 
during conversion.

Parameters:
- **path**: *str*  
  Path to save the model.
- **model_name**: *str*  
  The file name to be saved. 
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.

#### `ProgressiveSpatioTemporalGCNLearner.load`
```python
ProgressiveSpatioTemporalGCNLearner.load(self, path, model_name, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:
- **path**: *str*  
  Path of the model to be loaded.
- **model_name**: *str*  
  The file name to be loaded. 
- **verbose**: *bool, default=False*  
  If set to True, prints a message on success.


#### `ProgressiveSpatioTemporalGCNLearner.optimize`
```python
ProgressiveSpatioTemporalGCNLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*  
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the operations that have all constant inputs, with pre-computed constant nodes.


#### `ProgressiveSpatioTemporalGCNLearner.multi_stream_eval`
```python
ProgressiveSpatioTemporalGCNLearner.multi_stream_eval(self, dataset, scores, data_filename,
                                                      labels_filename, skeleton_data_type,
                                                      verbose, silent)
```
This method is used to ensemble the classification results of the model on two or more data streams like joints, bones and motions. 
It returns the top-k classification performance of ensembled model. 

Parameters:
- **dataset**: *object*  
  Object that holds the dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **score**: *list*  
  A list of score arrays. Each array in the list contains the evaluation results for a data stream.
- **data_filename**: *str, default='val_joints.npy'*  
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **labels_filename**: *str, default='val_labels.pkl'*  
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.
- **skeleton_data_type**: *str {'joint', 'bone', 'motion'}, default='joint'*  
  The data stream that should be used for training and evaluation. 
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*  
  If set to True, enables the maximum verbosity.


#### `ProgressiveSpatioTemporalGCNLearner.download`
```python
@staticmethod
ProgressiveSpatioTemporalGCNLearner.download(self, path, mode, verbose, url, file_name)
```

Download utility for various skeleton-based action recognition components. Downloads files depending on mode and
saves them in the path provided. It supports downloading:
1. the pretrained weights for stgcn, tagcn and stbln models. 
2. a dataset containing one or more skeleton sequences and its labels.  

Parameters:
- **path**: *str, default=None*  
  Local path to save the files, defaults to self.parent_dir if None.
- **mode**: *str, default="pretrained"*  
  What file to download, can be one of "pretrained", "train_data", "val_data", "test_data"
- **verbose**: *bool, default=False*  
  Whether to print messages in the console.
- **url**: *str, default=OpenDR FTP URL*  
  URL of the FTP server.
- **file_name**: *str*  
  The name of the file containing the pretrained model. 


#### Examples

* **Finding an optimized spatio-temporal GCN architecture based on training dataset defined as an `ExternalDataset`**.  
  The training and evaluation dataset should be present in the path provided, along with the labels file.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  from OpenDR.perception.skeleton_based_action_recognition.pstgcn_learner import ProgressiveSpatioTemporalGCNLearner
  from OpenDR.engine.datasets import ExternalDataset
  training_dataset = ExternalDataset(path='./data/preprocessed_nturgbd/xview', dataset_type='NTURGBD')
  validation_dataset = ExternalDataset(path='./data/preprocessed_nturgbd/xview', dataset_type='NTURGBD')
  
  pstgcn_learner = ProgressiveSpatioTemporalGCNLearner(temp_path='./parent_dir',
                                                       batch_size=64, epochs=65,
                                                       checkpoint_after_iter=10, val_batch_size=128,
                                                       dataset_name='nturgbd_cv', experiment_name='pstgcn_nturgbd',
                                                       blocksize=20, numblocks=1, numlayers=1, topology=[],
                                                       layer_threshold=1e-4, block_threshold=1e-4)
  
  pstgcn_learner.network_builder(dataset=training_dataset, val_dataset=validation_dataset,
                                 train_data_filename='train_joints.npy',
                                 train_labels_filename='train_labels.pkl',
                                 val_data_filename="val_joints.npy",
                                 val_labels_filename="val_labels.pkl",
                                 skeleton_data_type='joint')
  
  pstgcn_learner.save(path='./saved_models/pstgcn_nturgbd_cv_checkpoints', model_name='test_pstgcn')
  ```

* **Inference on a test skeleton sequence**
  ```python
  import numpy
  from OpenDR.perception.skeleton_based_action_recognition.pstgcn_learner import ProgressiveSpatioTemporalGCNLearner
  pstgcn_learner = ProgressiveSpatioTemporalGCNLearner(temp_path='./parent_dir',
                                                       batch_size=64, epochs=65,
                                                       checkpoint_after_iter=10, val_batch_size=128,
                                                       dataset_name='nturgbd_cv', experiment_name='pstgcn_nturgbd',
                                                       blocksize=20, numblocks=1, numlayers=1, topology=[],
                                                       layer_threshold=1e-4, block_threshold=1e-4)
  
  # Download the default pretrained stgcn model in the parent_dir
  pstgcn_learner.download(
            mode="pretrained", path='./parent_dir/pretrained_models', file_name='pretrained_pstgcn')  
  
  pstgcn_learner.load('./parent_dir/pretrained_models', model_name='pretrained_stgcn')
  test_data_path = pstgcn_learner.download(mode="test_data")  # Download a test data 
  test_data = numpy.load(test_data_path)
  action_category = pstgcn_learner.infer(test_data)
  
  ```
  
* **Optimization example for a previously trained model.**
  Inference can be run with the trained model after running self.optimize.
  ```python
  from OpenDR.perception.skeleton_based_action_recognition.pstgcn_learner import ProgressiveSpatioTemporalGCNLearner

  pstgcn_learner = ProgressiveSpatioTemporalGCNLearner(temp_path='./parent_dir',
                                                      batch_size=64, epochs=65,
                                                      checkpoint_after_iter=10, val_batch_size=128,
                                                      dataset_name='nturgbd_cv', experiment_name='pstgcn_nturgbd',
                                                      blocksize=20, numblocks=1, numlayers=1, topology=[],
                                                      layer_threshold=1e-4, block_threshold=1e-4)
  pstgcn_learner.download(
            mode="pretrained", path='./parent_dir/pretrained_models', file_name='pretrained_pstgcn')  
  
  pstgcn_learner.load(path='./parent_dir/pretrained_models', file_name='pretrained_pstgcn')
  pstgcn_learner.optimize(do_constant_folding=True)
  pstgcn_learner.save(path='./parent_dir/optimized_model', model_name='optimized_pstgcn')
  ```



## References

<a id="1">[1]</a> 
[Yan, S., Xiong, Y., & Lin, D. (2018, April). Spatial temporal graph convolutional networks for skeleton-based action 
recognition. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).](
https://arxiv.org/abs/1609.02907)

<a id="2">[2]</a> 
[Heidari, N., & Iosifidis, A. (2020). Temporal Attention-Augmented Graph Convolutional Network for Efficient Skeleton-
Based Human Action Recognition. arXiv preprint arXiv: 2010.12221.](https://arxiv.org/abs/2010.12221)

<a id="3">[3]</a> 
[Heidari, N., & Iosifidis, A. (2020). On the spatial attention in Spatio-Temporal Graph Convolutional Networks for 
skeleton-based human action recognition. arXiv preprint arXiv: 2011.03833.](https://arxiv.org/abs/2011.03833)

<a id="4">[4]</a> 
[Heidari, N., & Iosifidis, A. (2020). Progressive Spatio-Temporal Graph Convolutional Network for Skeleton-Based Human 
Action Recognition. arXiv preprint arXiv:2011.05668.](https://arxiv.org/pdf/2011.05668.pdf)

<a id="5">[5]</a> 
[Shahroudy, A., Liu, J., Ng, T. T., & Wang, G. (2016). Ntu rgb+ d: A large scale dataset for 3d human activity analysis.
 In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1010-1019).](
 https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html)

<a id="6">[6]</a>
[Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., ... & Zisserman, A. (2017). 
The kinetics human action video dataset. arXiv preprint arXiv:1705.06950.](https://arxiv.org/pdf/1705.06950.pdf) 

<a id="7">[7]</a>
[Cao, Z., Simon, T., Wei, S. E., & Sheikh, Y. (2017). Realtime multi-person 2d pose estimation using part affinity 
fields. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7291-7299).](
https://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.html)