## landmark_based_facial_expression_recognition module

The *landmark_based_facial_expression_recognition* module contains the *ProgressiveSpatioTemporalBLNLearner* class, which inherits from the abstract class *Learner*.

### Class ProgressiveSpatioTemporalBLNLearner
Bases: `engine.learners.Learner`

The *ProgressiveSpatioTemporalBLNLearner* class is an implementation of the proposed method PST-BLN [[6]](#6) for landmark-based facial expression recognition.
It finds an optimized and data dependant spatio-temporal bilinear network topology for landmark-based facial expression recognition.
The [ProgressiveSpatioTemporalBLNLearner](/src/opendr/perception/facial_expression_recognition/landmark_based_facial_expression_recognition/progressive_spatio_temporal_bln_learner.py) class has the
following public methods:


#### `ProgressiveSpatioTemporalBLNLearner` constructor
```python
ProgressiveSpatioTemporalBLNLearner(self, lr, batch_size, optimizer_name, lr_schedule,
                                    checkpoint_after_iter, checkpoint_load_iter, temp_path,
                                    device, num_workers, epochs, experiment_name,
                                    device_indices, val_batch_size, drop_after_epoch,
                                    start_epoch, dataset_name, num_class, num_point,
                                    num_person, in_channels,
                                    block_size, num_blocks, num_layers, topology,
                                    layer_threshold, block_threshold)
```

Constructor parameters:

- **lr**: *float, default=0.1*\
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=128*\
  Specifies number of samples to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **optimizer_name**: *str {'sgd', 'adam'}, default='sgd'*
  Specifies the optimizer type that should be used.
- **lr_schedule**: *str, default=' '*\
  Specifies the learning rate scheduler.
- **checkpoint_after_iter**: *int, default=0*\
  Specifies per how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default='temp'*\
  Specifies a path where the algorithm saves the checkpoints and onnx optimized model (if needed).
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **num_workers**: *int, default=32*\
  Specifies the number of workers to be used by the data loader.
- **epochs**: *int, default=400*\
  Specifies the number of epochs the training should run for.
- **experiment_name**: *str, default='pstbln_casia'*\
  String name to attach to checkpoints.
- **device_indices**: *list, default=[0]*\
  List of GPU indices to be used if the device is 'cuda'.
- **val_batch_size**: *int, default=128*\
  Specifies number of samples to be bundled up in a batch during evaluation. This heavily affects memory usage, adjust according to your system.
- **drop_after_epoch**: *list, default=[400]*\
  List of epoch numbers in which the optimizer drops the learning rate.
- **start_epoch**: *int, default=0*\
  Specifies the starting epoch number for training.
- **dataset_name**: *str {'AFEW', 'CASIA', 'CK+'}, default='CASIA'*\
  Specifies the name of dataset that is used for training and evaluation.
- **num_class**: *int, default=6*\
  Specifies the number of classes for the dataset.
- **num_point**: *int, default=309*\
  Specifies the number of facial landmarks in each image.
- **num_person**: *int, default=1*\
  Specifies the number of faces in each frame.
- **in_channels**: *int, default=2*\
  Specifies the number of input channels for each landmark.
- **block_size**: *int, default=5*\
  Specifies the number of output channels (or neurons) that are added to each layer of the network at each progression iteration.
- **num_blocks**: *int, default=100*\
  Specifies the maximum number of blocks that are added to each layer of the network at each progression iteration.
- **num_layers**: *int, default=10*\
  Specifies the maximum number of layers that are built for the network.
- **topology**: *list, default=[]*\
  Specifies the initial topology of the network. The default is set to [], since the method gets an empty network as input and builds it progressively.
- **layer_threshold**: *float, default=1e-4*\
  Specifies the threshold which is used by the method to identify when it should stop adding new layers.
- **block_threshold**: *float, default=1e-4*\
  Specifies the threshold which is used by the model to identify when it should stop adding new blocks in each layer.


#### `ProgressiveSpatioTemporalBLNLearner.fit`
```python
ProgressiveSpatioTemporalBLNLearner.fit(self, dataset, val_dataset, logging_path, silent, verbose,
                                        momentum, nesterov, weight_decay, monte_carlo_dropout,
                                        mcdo_repeats, train_data_filename,
                                        train_labels_filename, val_data_filename,
                                        val_labels_filename)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **val_dataset**: *object*\
  Object that holds the validation dataset.
- **logging_path**: *str, default=''*\
  Path to save TensorBoard log files and the training log files.
  If set to None or '', TensorBoard logging is disabled and no log file is created.
- **silent**: *bool, default=False*\
  If set to True, disables all printing of training progress reports and other information to STDOUT.
- **verbose**: *bool, default=True***\
  If set to True, enables the maximum verbosity.
- **momentum**: *float, default=0.9*\
  Specifies the momentum value for optimizer.
- **nesterov**: *bool, default=True***\
  If set to true, the optimizer uses Nesterov's momentum.
- **weight_decay**: *float, default=0.0001***\
  Specifies the weight_decay value of the optimizer.
- **monte_carlo_dropout**: *bool, default=True***\
  If set to True, enables the Monte Carlo Dropout in inference.
- **mcdo_repeats**: *int, default=100***\
  Specifies the number of times that inference is repeated for Monte Carlo Dropout.
- **train_data_filename**: *str, default='train.npy'*\
  Filename that contains the training data.
  This file should be contained in the dataset path provided.
  Note that this is a file name, not a path.
- **train_labels_filename**: *str, default='train_labels.pkl'*\
  Filename of the labels .pkl file.
  This file should be contained in the dataset path provided.
- **val_data_filename**: *str, default='val.npy'*\
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **val_labels_filename**: *str, default='val_labels.pkl'*\
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.


#### `ProgressiveSpatioTemporalBLNLearner.eval`
```python
ProgressiveSpatioTemporalBLNLearner.eval(self, val_dataset, val_loader, epoch, monte_carlo_dropout,
                                         mcdo_repeats, silent, verbose,
                                         val_data_filename, val_labels_filename,
                                         save_score, wrong_file, result_file, show_topk)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:

- **val_dataset**: *object*\
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **val_loader**: *object, default=None*\
  Object that holds a Python iterable over the evaluation dataset.
  Object of `torch.utils.data.DataLoader` class.
- **epoch**: *int, default=0*\
  The training epoch in which the model is evaluated.
- **monte_carlo_dropout**: *bool, default=True***\
  If set to True, enables the Monte Carlo Dropout in inference.
- **mcdo_repeats**: *int, default=100***\
  Specifies the number of times that inference is repeated for Monte Carlo Dropout.
- **silent**: *bool, default=False*\
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*\
  If set to True, enables the maximum verbosity.
- **val_data_filename**: *str, default='val.npy'*\
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **val_labels_filename**: *str, default='val_labels.pkl'*\
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.
- **save_score**: *bool, default=False*\
  If set to True, it saves the classification score of all samples in differenc classes
  in a log file. Default to False.
- **wrong_file**: *str, default=None*\
  If set to True, it saves the results of wrongly classified samples. Default to False.
- **result_file**: *str, default=None*\
  If set to True, it saves the classification results of all samples. Default to False.
- **show_topk**: *list, default=[1, 5]*\
  Is set to a list of integer numbers defining the k in top-k accuracy. Default is set to [1,5].


#### `ProgressiveSpatioTemporalBLNLearner.init_model`
```python
ProgressiveSpatioTemporalBLNLearner.init_model(self)
```
This method is used to initialize the imported model and its loss function.


#### `ProgressiveSpatioTemporalBLNLearner.network_builder`
```python
ProgressiveSpatioTemporalBLNLearner.network_builder(self, dataset, val_dataset, monte_carlo_dropout,
                                                    mcdo_repeats, logging_path, train_data_filename,
                                                    train_labels_filename, val_data_filename,
                                                    val_labels_filename, verbose)
```
This method builds the network topology progressively.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset.
- **val_dataset**: *object*\
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.
- **monte_carlo_dropout**: *bool, default=True***\
  If set to True, enables the Monte Carlo Dropout in inference.
- **mcdo_repeats**: *int, default=100***\
  Specifies the number of times that inference is repeated for Monte Carlo Dropout.
- **logging_path**: *str, default=''***\
  path to save tensorboard log files. If set to None or '', tensorboard logging is disabled.
- **train_data_filename**: *str, default='train.npy'*\
  Filename that contains the training data.
  This file should be contained in the dataset path provided.
  Note that this is a file name, not a path.
- **train_labels_filename**: *str, default='train_labels.pkl'*\
  Filename of the labels .pkl file.
  This file should be contained in the dataset path provided.
- **val_data_filename**: *str, default='val.npy'*\
  Filename that contains the validation data.
  This file should be contained in the dataset path provided.
  Note that this is a filename, not a path.
- **val_labels_filename**: *str, default='val_labels.pkl'*\
  Filename of the validation labels .pkl file.
  This file should be contained in the dataset path provided.
- **verbose**: *bool, default=True***\
  Whether to print messages in the console.


#### `ProgressiveSpatioTemporalBLNLearner.infer`
```python
ProgressiveSpatioTemporalBLNLearner.infer(self, facial_landmarks_batch, monte_carlo_dropout, mcdo_repeats)
```
This method is used to perform action recognition on a sequence of landmarks.
It returns the action category as an object of `engine.target.Category` if a proper input object `engine.data.SkeletonSequence` is given.

Parameters:
- **facial_landmarks_batch**: *object***
  Object of type `engine.data.SkeletonSequence`.
- **monte_carlo_dropout**: *bool, default=True***
  If set to True, enables the Monte Carlo Dropout in inference.
- **mcdo_repeats**: *int, default=100***
  Specifies the number of times that inference is repeated for Monte Carlo Dropout.

#### `ProgressiveSpatioTemporalBLNLearner.save`
```python
ProgressiveSpatioTemporalBLNLearner.save(self, path, model_name, verbose)
```
This method is used to save a trained model.
Provided with the path "/my/path" (absolute or relative), it creates the "path" directory, if it does not already
exist. Inside this folder, the model is saved as "model_name.pt" and the metadata file as "model_name.json". If the directory
already exists, the "model_name.pt" and "model_name.json" files are overwritten.

If [`self.optimize`](#ProgressiveSpatioTemporalBLNLearner.optimize) was run previously, it saves the optimized ONNX model in
a similar fashion with an ".onnx" extension, by copying it from the self.temp_path it was saved previously
during conversion.

Parameters:

- **path**: *str*\
  Path to save the model.
- **model_name**: *str*\
  The file name to be saved.
- **verbose**: *bool, default=True*\
  If set to True, prints a message on success.

#### `ProgressiveSpatioTemporalBLNLearner.load`
```python
ProgressiveSpatioTemporalBLNLearner.load(self, path, model_name, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.
- **model_name**: *str*\
  The file name to be loaded.
- **verbose**: *bool, default=True*\
  If set to True, prints a message on success.


#### `ProgressiveSpatioTemporalBLNLearner.optimize`
```python
ProgressiveSpatioTemporalBLNLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:

- **do_constant_folding**: *bool, default=False*\
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.
  Constant-folding optimization will replace some of the operations that have all constant inputs, with pre-computed constant nodes.


#### `ProgressiveSpatioTemporalBLNLearner.download`
```python
@staticmethod
ProgressiveSpatioTemporalBLNLearner.download(self, path, mode, verbose, url)
```
Downloads files depending on mode and saves them in the path provided. It supports downloading train, validation and test dataset containing one or more landmark sequences and its labels.

Parameters:

- **path**: *str, default=None*\
  Local path to save the files, defaults to `self.parent_dir` if None.
- **mode**: *str, default="train_data"*\
  What file to download, can be one of "pretrained", "train_data", "val_data", "test_data"
- **verbose**: *bool, default=True*\
  Whether to print messages in the console.
- **url**: *str, default=opendr FTP URL*\
  URL of the FTP server.



#### Data preparation
  Download the [AFEW](https://cs.anu.edu.au/few/AFEW.html) [[1]](https://www.computer.org/csdl/magazine/mu/2012/03/mmu2012030034/13rRUxjQyrW), [[2]](https://dl.acm.org/doi/abs/10.1145/2663204.2666275), and [CK+](https://www.pitt.edu/~emotion/ck-spread.htm) [[3]](https://ieeexplore.ieee.org/abstract/document/5543262), [[4]](https://ieeexplore.ieee.org/abstract/document/840611) and [Oulu-CASIA](https://www.oulu.fi/cmvs/node/41316) [[5]](https://www.sciencedirect.com/science/article/pii/S0262885611000515) datasets.
  In order to extract facial landmarks from the images, you need to download a pretrained landmark extractor model.
  We used Dlib's landmark extractor which can be downloaded from [here](http://dlib.net/face_landmark_detection.py.html).
  Please note that these datasets and the landmark extractor model cannot be used for any commercial purposes.

  ##### AFEW data preparation:
  AFEW dataset consists of a set of video clips collected from movies with actively moving faces in different illumination and environmental conditions.
  The following preprocessing steps are needed to generate the appropriate landmark data for the method.
  - Convert the .avi videos to .mp4 format and then extract the video frames using the following function:

  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import frame_extractor
  python3 frame_extractor.py --video_folder ./data/AFEW_videos/ --frames_folder ./data/AFEW/
  ```
  You need to specify the path of the videos as `--video_folder` and the path of the extracted frames data as `--frames_folder`.

  - Place the downloaded landmark-extractor model in the data directory and extract the facial landmarks from the extracted frames by running the following script:

  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import landmark_extractor
  python3 landmark_extractor.py --dataset_name AFEW --shape_predictor ./data/shape_predictor_68_face_landmarks.dat --frames_folder ./data/AFEW/ --landmark_folder ./data/AFEW_landmarks/
  ```
  You need to specify the path of the landmark extractor as `--shape_predictor` and the path of the extracted frames and extracted landmarks as `--frames_folder` and `--landmark_folder`.

 - After extracting the facial landmarks for each category, run the following script for data preprocessing and augmentation:

  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import AFEW_data_gen
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import data_augmentation
  python3 AFEW_data_gen.py --landmark_folder  ./data/AFEW_landmarks/ --data_folder ./data/AFEW_data/
  python3 data_aumentation.py --data_folder ./data/AFEW_data/ --aug_data_folder ./data/AFEW_aug_data/
  ```
  The preprocessed augmented data will be saved in the `--aug_data_folder` path.
  After generating the preprocessed facial landmark data, generate the facial graph edges data as follows:
  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import gen_facial_muscles_data
  python3 gen_facial_muscles_data.py --dataset_name AFEW --landmark_data_folder ./data/AFEW_aug_data/ --muscle_data_folder ./data/muscle_data/
  ```


##### CK+ data preparation:
  CK+ dataset consists of a set of image sequences starting from a neutral expression to peak expression and the expressions are performed by different subjects.
  We select the first frame and the last three frames (including the peak expression) of each sequence for landmark extraction.
  In this dataset, only a subset of image sequences are labeled.
  The first step in the data preparation is to separate the labeled data for each subject, and place each sample in a folder named by its class label.
  - Extract the facial landmarks and generate the preprocessed train and test data for 10-fold cross validation using the following script:

  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import CASIA_CK+_data_gen
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import landmark_extractor
  python3 landmark_extractor.py --dataset_name CK+ --shape_predictor ./data/shape_predictor_68_face_landmarks.dat --frames_folder ./data/CK+/ --landmark_folder ./data/CK+_landmarks/
  python3 CASIA_CK+_data_gen.py --dataset_name CK+ --landmark_folder  ./data/CK+_landmarks/ --output_folder ./data/CK+_10fold/
  ```
  - After generating the preprocessed facial landmark data, generate the facial muscle data as follows:
  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import gen_facial_muscles_data
  python3 gen_facial_muscles_data.py --dataset_name CK+ --landmark_data_folder ./data/CK+_10fold/ --muscle_data_folder ./data/muscle_data/
  ```

##### Oulu-CASIA data preparation:
  Oulu-CASIA dataset also consists of a set of image sequences starting from a neutral expression to peak expression and the expressions are performed by different subjects.
  We used the image sequences captured by the VIS system under NI illumination and we select the first frame and the last three frames (including the peak expression) of each sequence for landmark extraction.
  - Extract the facial landmarks and generate the preprocessed train and test data for 10-fold cross validation using the following script:
  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import CASIA_CK+_data_gen
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import landmark_extractor
  python3 landmark_extractor.py --dataset_name CASIA --shape_predictor ./data/shape_predictor_68_face_landmarks.dat --frames_folder ./data/CASIA/ --landmark_folder ./data/CASIA_landmarks/
  python3 CASIA_CK+_data_gen.py --dataset_name CASIA --landmark_folder  ./data/CASIA_landmarks/ --output_folder ./data/CASIA_10fold/
  ```
  - After generating the preprocessed facial landmark data, generate the facial muscle data as follows:
  ```python
  from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.algorithm.datasets import gen_facial_muscles_data
  python3 gen_facial_muscles_data.py --dataset_name CASIA --landmark_data_folder ./data/CASIA_10fold/ --muscle_data_folder ./data/muscle_data/
  ```

#### Examples

* **Finding an optimized spatio-temporal BLN architecture based on training dataset defined as an `ExternalDataset`**.
  The training and evaluation dataset should be present in the path provided, along with the labels file.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  from opendr.perception.facial_expression_recognition import ProgressiveSpatioTemporalBLNLearner
  from opendr.engine.datasets import ExternalDataset
  training_dataset = ExternalDataset(path='./data/AFEW_aug_data', dataset_type='AFEW')
  validation_dataset = ExternalDataset(path='./data/AFEW_aug_data', dataset_type='AFEW')

  pstbln_learner = ProgressiveSpatioTemporalBLNLearner(temp_path='./parent_dir',
                                                       batch_size=64, epochs=400,
                                                       checkpoint_after_iter=10, val_batch_size=128,
                                                       dataset_name='AFEW', experiment_name='pstbln_afew',
                                                       block_size=5, num_blocks=10, num_layers=10, topology=[],
                                                       layer_threshold=1e-4, block_threshold=1e-4)

  pstbln_learner.network_builder(dataset=training_dataset, val_dataset=validation_dataset,
                                 monte_carlo_dropout=True, mcdo_repeats=100,
                                 logging_path='',
                                 train_data_filename='train.npy',
                                 train_labels_filename='train_labels.pkl',
                                 val_data_filename="val.npy",
                                 val_labels_filename="val_labels.pkl")

  pstbln_learner.save(path='./saved_models/pstbln_afew_checkpoints', model_name='test_pstbln')
  ```

* **Inference on a test landmark sequence**
  ```python
  import numpy
  from opendr.perception.facial_expression_recognition import ProgressiveSpatioTemporalBLNLearner
  pstbln_learner = ProgressiveSpatioTemporalBLNLearner(temp_path='./parent_dir',
                                                       batch_size=64, epochs=400,
                                                       checkpoint_after_iter=10, val_batch_size=128,
                                                       dataset_name='AFEW', experiment_name='pstbln_afew',
                                                       block_size=5, num_blocks=10, num_layers=10, topology=[],
                                                       layer_threshold=1e-4, block_threshold=1e-4)

  # Download the test data and place it in the parent_dir
  Test_DATASET_PATH = pstbln_learner.download(mode="test_data", path="./data/afew")
  test_data = numpy.load(Test_DATASET_PATH)[0:1]
  expression_category = pstbln_learner.infer(test_data)
  ```

* **Optimization example for a previously trained model.**
  Inference can be run with the trained model after running self.optimize.
  ```python
  from opendr.perception.facial_expression_recognition import ProgressiveSpatioTemporalBLNLearner

  pstbln_learner = ProgressiveSpatioTemporalBLNLearner(temp_path='./parent_dir',
                                                      batch_size=64, epochs=400,
                                                      checkpoint_after_iter=10, val_batch_size=128,
                                                      dataset_name='AFEW', experiment_name='pstbln_afew',
                                                      block_size=5, num_blocks=10, num_layers=10, topology=[],
                                                      layer_threshold=1e-4, block_threshold=1e-4)

  pstbln_learner.load(path='./parent_dir/pretrained_models', file_name='pretrained_pstbln')
  pstbln_learner.optimize(do_constant_folding=True)
  pstbln_learner.save(path='./parent_dir/optimized_model', model_name='optimized_pstbln')
  ```


#### Performance Evaluation

The tests were conducted on the following computational devices:
- Intel(R) Xeon(R) Gold 6230R CPU on server
- Nvidia Jetson TX2
- Nvidia Jetson Xavier AGX
- Nvidia RTX 2080 Ti GPU on server with Intel Xeon Gold processors

Inference time is measured as the time taken to transfer the input to the model (e.g., from CPU to GPU), run inference using the algorithm, and return results to CPU.
The PST-BLN model is implemented in *ProgressiveSpatioTemporalBLNLearner*.

Note that the model receives each input sample as a sequence of 150 graphs built by facial landmarks as nodes and the connections between them as edges.
The facial landmarks are extracted by Dlib library as a preprocessing step, and the landmark extraction process is not involved in this benchmarking.
The model is evaluated on the AFEW dataset which contains Acted Facial Expression in the Wild video clips captured from movies.
We report speed (single sample per inference), as the mean of 100 runs, of the optimized ST-BLN model found by PST-BLN algorithm.
The noted memory is the maximum allocated memory on GPU during inference.
 
Prediction accuracy on AFEW dataset, parameter count and maximum allocated memory of learner's inference are reported in the following table:

| Method  | Acc. (%) | Params (M) | Mem. (MB) | 
|---------|----------|------------|-----------|
| PST-BLN | 33.33    | 0.01       | 76.41     | 

The speed (evaluations/second) of the learner's inference on various computational devices is:

| Method  | CPU   | Jetson TX2 | Jetson Xavier  | RTX 2080 Ti | 
|---------|-------|------------|----------------|-------------|
| PST-BLN | 8.05  | 3.81       | 14.27          | 125.17      | 


Energy (Joules) of both learners’ inference on embedded devices is shown in the following: 

| Method  | Jetson TX2 | Jetson Xavier | 
|---------|------------|---------------|
| PST-BLN | 5.33       | 1.12          | 


The platform compatibility evaluation is also reported below:

| Platform  | Compatibility Evaluation |
| ----------------------------------------------|--------------------------|
| x86 - Ubuntu 20.04 (bare installation - CPU)  | :heavy_check_mark:       |
| x86 - Ubuntu 20.04 (bare installation - GPU)  | :heavy_check_mark:       |
| x86 - Ubuntu 20.04 (pip installation)         | :heavy_check_mark:       |
| x86 - Ubuntu 20.04 (CPU docker)               | :heavy_check_mark:       |
| x86 - Ubuntu 20.04 (GPU docker)               | :heavy_check_mark:       |
| NVIDIA Jetson TX2                             | :heavy_check_mark:       |
| NVIDIA Jetson Xavier AGX                      | :heavy_check_mark:       |



## References

<a id="1">[1]</a>
[Dhall, Abhinav, et al. "Collecting large, richly annotated facial-expression databases from movies." IEEE Annals of the History of Computing 19.03 (2012): 34-41.](
https://www.computer.org/csdl/magazine/mu/2012/03/mmu2012030034/13rRUxjQyrW)

<a id="2">[2]</a>
[Dhall, Abhinav, et al. "Emotion recognition in the wild challenge 2014: Baseline, data and protocol." Proceedings of the 16th international conference on multimodal interaction. 2014.](https://dl.acm.org/doi/abs/10.1145/2663204.2666275)

<a id="3">[3]</a>
[Lucey, Patrick, et al. "The extended cohn-kanade dataset (ck+): A complete dataset for action unit and emotion-specified expression." 2010 ieee computer society conference on computer vision and pattern recognition-workshops. IEEE, 2010.](https://ieeexplore.ieee.org/abstract/document/5543262)

<a id="4">[4]</a>
[Kanade, Takeo, Jeffrey F. Cohn, and Yingli Tian. "Comprehensive database for facial expression analysis." Proceedings Fourth IEEE International Conference on Automatic Face and Gesture Recognition (Cat. No. PR00580). IEEE, 2000.](https://ieeexplore.ieee.org/abstract/document/840611)

<a id="5">[5]</a>
[Zhao, Guoying, et al. "Facial expression recognition from near-infrared videos." Image and Vision Computing 29.9 (2011): 607-619.](
https://www.sciencedirect.com/science/article/pii/S0262885611000515)

<a id="6">[6]</a>
[Heidari, Negar, and Alexandros Iosifidis. "Progressive Spatio-Temporal Bilinear Network with Monte Carlo Dropout for Landmark-based Facial Expression Recognition with Uncertainty Estimation." arXiv preprint arXiv:2106.04332 (2021).](https://arxiv.org/abs/2106.04332)
