## image_based_facial_emotion_estimation module

The *image_based_facial_emotion_estimation* module contains the *FacialEmotionLearner* class, which inherits from the abstract class *Learner*.

### Class FacialEmotionLearner
Bases: `engine.learners.Learner`

The *FacialEmotionLearner* class is an implementation of the state-of-the-art method ESR [[1]](#1) for efficient facial feature learning with wide ensemble-based convolutional neural networks.
An ESR consists of two building blocks.
(1) The base of the network is an array of convolutional layers for low- and middle-level feature learning.
(2) These informative features are then shared with independent convolutional branches that constitute the ensemble.
From this point, each branch can learn distinctive features while competing for a common resource - the shared layers.
The [FacialEmotionLearner](/src/opendr/perception/facial_expression_recognition/image_based_facial_emotion_estimation/facial_emotion_learner.py) class has the following public methods:


#### `FacialEmotionLearner` constructor
```python
FacialEmotionLearner(self, lr, batch_size, temp_path, device, device_ind, validation_interval,
                     max_training_epoch, momentum, ensemble_size, base_path_experiment, name_experiment, dimensional_finetune, categorical_train,
                     base_path_to_dataset, max_tuning_epoch, diversify)
```

Constructor parameters:

- **lr**: *float, default=0.1*\
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=32*\
  Specifies number of samples to be bundled up in a batch during training.
  This heavily affects memory usage, adjust according to your system.
- **temp_path**: *str, default='temp'*\
  Specifies a path where the algorithm saves the checkpoints and onnx optimized model (if needed).
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **device_ind**: *list, default=[0]*\
  List of GPU indices to be used if the device is 'cuda'.
- **validation_interval**: *int, default=1*\
  Specifies the validation interval.
- **max_training_epoch**: *int, default=2*\
  Specifies the maximum number of epochs the training should run for.
- **momentum**: *float, default=0.9*\
  Specifies the momentum value used for optimizer.
- **ensemble_size**: *int, default=9*\
  Specifies the number of ensemble branches in the model.
- **base_path_experiment**: *str, default='./experiments/'*\
  Specifies the path in which the experimental results will be saved.
- **name_experiment**: *str, default='esr_9'*\
  String name for saving checkpoints.
- **dimensional_finetune**: *bool, default=True*\
  Specifies if the model should be fine-tuned on dimensional data or not.
- **categorical_train**: *bool, default=False*\
  Specifies if the model should be trained on categorical data or not.
- **base_path_to_dataset**: *str, default=''./data/AffectNet''*\
  Specifies the dataset path.
- **max_tuning_epoch**: *int, default=1*\
  Specifies the maximum number of epochs the model should be finetuned on dimensional data.
- **diversity**: *bool, default=False*\
  Specifies if the learner diversifies the features of different branches or not.

#### `FacialEmotionLearner.fit`
```python
FacialEmotionLearner.fit(self)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.


#### `FacialEmotionLearner.eval`
```python
FacialEmotionLearner.eval(self, eval_type, current_branch_on_training)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:

- **eval_type**: *str, default='categorical'*\
  Specifies the type of data that model is evaluated on.
  It can be either categorical or dimensional data.
- **current_branch_on_training**: *int, default=0*\
  Specifies the index of trained branch which should be evaluated on validation data.


#### `FacialEmotionLearner.init_model`
```python
FacialEmotionLearner.init_model(self, num_branches)
```

This method is used to initialize the model.

Parameters:

- **num_branches**: *int*\
  Specifies the number of ensemble branches in the model. ESR_9 model is built by 9 branches by default.

#### `FacialEmotionLearner.infer`
```python
FacialEmotionLearner.infer(self, input_batch)
```

This method is used to perform inference on an image or a batch of images.
It returns dimensional emotion results and also the categorical emotion results as an object of `engine.target.Category` if a proper input object `engine.data.Image` is given.

Parameters:

- **input_batch**: *object***
  Object of type `engine.data.Image`. It also can be a list of Image objects, or a Torch tensor which will be converted to Image object.

#### `FacialEmotionLearner.save`
```python
FacialEmotionLearner.save(self, state_dicts, base_path_to_save_model)
```
This method is used to save a trained model.
Provided with the path (absolute or relative), it creates the "path" directory, if it does not already exist.
Inside this folder, the model is saved as "model_name.pt" and the metadata file as "model_name.json". If the directory already exists, the "model_name.pt" and "model_name.json" files are overwritten.

If [`self.optimize`](#FacialEmotionLearner.optimize) was run previously, it saves the optimized ONNX model in a similar fashion with an ".onnx" extension, by copying it from the self.temp_path it was saved previously during conversion.

Parameters:

- **state_dicts**: *object*\
  Object of type Python dictionary containing the trained model weights.
- **base_path_to_save_model**: *str*\
  Specifies the path in which the model will be saved.

#### `FacialEmotionLearner.load`
```python
FacialEmotionLearner.load(self, ensemble_size, path_to_saved_network, file_name_base_network,
                        file_name_conv_branch, fix_backbone)
```

Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:

- **ensemble_size**: *int, default=9*\
  Specifies the number of ensemble branches in the model for which the pretrained weights should be loaded.
- **path_to_saved_network**: *str, default="./trained_models/esr_9"*\
  Path of the model to be loaded.
- **file_name_base_network**: *str, default="Net-Base-Shared_Representations.pt"*\
  The file name of the base network to be loaded.
- **file_name_conv_branch**: *str, default="Net-Branch_{}.pt"*\
  The file name of the ensemble branch network to be loaded.
- **fix_backbone**: *bool*\
  If true, all the model weights except the classifier are fixed so that the last layers' weights are fine-tuned on dimensional data.
  Otherwise, all the model weights will be trained from scratch.


#### `FacialEmotionLearner.optimize`
```python
FacialEmotionLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:

- **do_constant_folding**: *bool, default=False*\
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.


#### `FacialEmotionLearner.download`
```python
@staticmethod
FacialEmotionLearner.download(self, path, mode, url)
```

Downloads data and saves them in the path provided.

Parameters:

- **path**: *str, default=None*\
  Local path to save the files, defaults to `self.temp_dir` if None.
- **mode**: *str, default="data"*\
  What file to download, can be "data".
- **url**: *str, default=opendr FTP URL*\
  URL of the FTP server.


#### Data preparation
  Download the [AffectNet](http://mohammadmahoor.com/affectnet/) [[2]](https://www.computer.org/csdl/magazine/mu/2012/03/mmu2012030034/13rRUxjQyrW) dataset, and organize it in the following structure:
  ```
  AffectNet/
      Training_Labeled/
          0/
          1/
          ...
          n/
      Training_Unlabeled/
          0/
          1/
          ...
          n/
      Validation/
          0/
          1/
          ...
          n/
  ```
  In order to do that, you need to run the following function:
  ```python
  from opendr.perception.facial_expression_recognition.image_based_facial_emotion_estimation.algorithm.utils import datasets
  datasets.pre_process_affect_net(base_path_to_images, base_path_to_annotations, base_destination_path, set_index)
  ```
  This pre-processes the AffectNet dataset by cropping and resizing the images into 96 x 96 pixels, and organizing them in folders with 500 images each.
  Each image is renamed to follow the pattern "[id][emotion_idx][valence times 1000]_[arousal times 1000].jpg".

#### Pre-trained models

The pretrained models on AffectNet Categorical dataset are provided by [[1]](#1) which can be found [here](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/tree/master/model/ml/trained_models/esr_9).
**Please note that the pretrained weights cannot be used for commercial purposes.**

#### Examples

* **Train the ensemble model on AffectNet Categorical dataset and then fine-tune it on the AffectNet dimensional dataset**
  The training and evaluation dataset should be present in the path provided.
  The `batch_size` argument should be adjusted according to available memory.

  ```python
  from opendr.perception.facial_expression_recognition import FacialEmotionLearner

  learner = FacialEmotionLearner(device="cpu", temp_path='./tmp',
                                 batch_size=2, max_training_epoch=1, ensemble_size=1,
                                 name_experiment='esr_9', base_path_experiment='./experiments/',
                                 lr=1e-1, categorical_train=True, dimensional_finetune=True,
                                 base_path_to_dataset='./data', max_tuning_epoch=1)
  learner.fit()
  learner.save(state_dicts=learner.model.to_state_dict(),
               base_path_to_save_model=learner.base_path_experiment,
               current_branch_save=8)
  ```

* **Inference on a batch of images**
  ```python
  from opendr.perception.facial_expression_recognition import FacialEmotionLearner
  from torch.utils.data import DataLoader

  learner = FacialEmotionLearner(device="cpu", temp_path='./tmp', 
                                 batch_size=2, max_training_epoch=1, ensemble_size=1,
                                 name_experiment='esr_9', base_path_experiment='./experiments/',
                                 lr=1e-1, categorical_train=True, dimensional_finetune=True,
                                 base_path_to_dataset='./data', max_tuning_epoch=1)

  # Download the validation data
  dataset_path = learner.download(mode='data')
  val_data = datasets.AffectNetCategorical(idx_set=2,
                                           max_loaded_images_per_label=2,
                                           transforms=None,
                                           is_norm_by_mean_std=False,
                                           base_path_to_affectnet=learner.dataset_path)

  val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=8)
  batch = next(iter(val_loader))[0]
  learner.load(learner.ensemble_size, path_to_saved_network=learner.base_path_experiment, fix_backbone=True)
  ensemble_emotion_results, ensemble_dimension_results = learner.infer(batch[0])
  ```

* **Optimization example for a previously trained model**
  Inference can be run with the trained model after running self.optimize.
  ```python
  from opendr.perception.facial_expression_recognition import FacialEmotionLearner

  learner = FacialEmotionLearner(device="cpu", temp_path='./tmp',
                               batch_size=2, max_training_epoch=1, ensemble_size=1,
                               name_experiment='esr_9', base_path_experiment='./experiments/',
                               lr=1e-1, categorical_train=True, dimensional_finetune=True,
                               base_path_to_dataset='./data', max_tuning_epoch=1)


  learner.load(learner.ensemble_size, path_to_saved_network=learner.base_path_experiment, fix_backbone=True)
  learner.optimize(do_constant_folding=True)
  learner.save(path='./parent_dir/optimized_model', model_name='optimized_pstbln')
  ```


#### Performance Evaluation

The tests were conducted on the following computational devices:
- Intel(R) Xeon(R) Gold 6230R CPU on server
- Nvidia Jetson TX2
- Nvidia Jetson Xavier AGX
- Nvidia RTX 2080 Ti GPU on server with Intel Xeon Gold processors


Inference time is measured as the time taken to transfer the input to the model (e.g., from CPU to GPU), run inference using the algorithm, and return results to CPU.
The ESR and its extension diversified_ESR denoted as ESR*, which learns diversified feature representations to improve the model generalisation, are implemented in *FacialEmotionLearner*.
The ESR-n and ESR*-n denote the ESR and diversified-ESR models with #n ensemble branches, respectively

The model can receive either single images as input or a video, which can be captured by webcam, and perform the prediction frame-by-frame.

We report speed (single sample per inference) as the mean of 100 runs, and the energy (Joules) on embedded devices.
The noted memory is the maximum allocated memory on GPU during inference.

| Method       | Acc. (%) | Params (M) | Mem. (MB) | 
|--------------|----------|------------|-----------|
| ESR-9        | 87.17    | 20.35      | 402.99    | 
| ESR-15       | 88.59    | 33.67      | 455.61    | 
| ESR*-9       | 89.15    | 20.83      | 406.83    |
| ESR*-15      | 89.34    | 34.47      | 460.73    |

The inference speed (evaluations/second) of both learners on various computational devices are as follows:

| Method       | CPU   | Jetson TX2 | Jetson Xavier | RTX 2080 Ti | 
|--------------|-------|------------|---------------|-------------|
| ESR-9        | 22.23 | 27.08      | 28.79         | 117.91      | 
| ESR-15       | 13.86 | 17.76      | 18.17         | 91.78       | 
| ESR*-9       | 5.24  | 6.60       | 12.45         | 33.40       |
| ESR*-15      | 3.38  | 4.18       | 8.47          | 20.57       | 

Energy (Joules) of both learners’ inference on embedded devices is shown in the following: 

| Method  | Jetson TX2 | Jetson Xavier | 
|---------|------------|---------------|
| ESR-9   | 0.96       | 0.67          | 
| ESR-15  | 1.16       | 0.93          | 
| ESR*-9  | 3.38       | 1.41          |
| ESR*-15 | 6.26       | 2.51          | 




## References

<a id="1">[1]</a>
[Siqueira, Henrique, Sven Magg, and Stefan Wermter. "Efficient facial feature learning with wide ensemble-based convolutional neural networks." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 04. 2020.](
https://ojs.aaai.org/index.php/AAAI/article/view/6037)

<a id="2">[2]</a>
[Mollahosseini, Ali, Behzad Hasani, and Mohammad H. Mahoor. "Affectnet: A database for facial expression, valence, and arousal computing in the wild." IEEE Transactions on Affective Computing 10.1 (2017): 18-31.](
https://ieeexplore.ieee.org/abstract/document/8013713)
