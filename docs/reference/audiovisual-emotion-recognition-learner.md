## audiovisual_emotion_learner module

The *audiovisual_emotion_learner* module contains the *AudiovisualEmotionLearner* class, which inherits from the abstract class *Learner*.

### Class AudiovisualEmotionLearner
Bases: `opendr.engine.learners.Learner`

The *AudiovisualEmotionLearner* class provides an implementation of audiovisual emotion recognition method using audio and video (frontal face) inputs.
The implementation follows the method described in ['Self-attention fusion for audiovisual emotion recognition with incomplete data'](https://arxiv.org/abs/2201.11095).
Three fusion methods are provided.

AudiovisualEmotionLearner relies on EfficientFace model implementation [1].
Parts of training pipeline are modified from [2].

The [AudiovisualEmotionLearner](/src/opendr/perception/multimodal_human_centric/audiovisual_emotion_learner/avlearner.py) class has the following public methods:

#### `AudiovisualEmotionLearner` constructor
```python
AudiovisualEmotionLearner(self, num_class, seq_length, fusion, mod_drop, pretr_ef, lr, lr_steps, momentum, dampening, weight_decay, iters, batch_size, n_workers, checkpoint_after_ietr, checkpoint_load_iter, temp_path, device)
```

**Parameters**:

- **num_class**: *int, default=8*\
  Specifies the number of classes.

- **seq_length**: *int, default=15*\
  Length of frame sequence representing a video.

- **fusion**: *str, default='ia'*\
  Modality fusion method.
  Options are 'ia', 'it', 'lt', referring to 'intermediate attention', 'intermediate transformer', and 'late transformer' fusion.
  Refer [here](https://arxiv.org/abs/2201.11095) for details.

- **mod_dropout**: *str, default='zerodrop', {'nodrop', 'noisedrop', 'zerodrop'}*
  Modality dropout method.
  Refer to [here](https://arxiv.org/abs/2201.11095) for details.

- **pretr_ef**: *str, default=None*
  Checkpoint of the pre-trained EfficientFace model that is used to initialize the weights of the vision backbone.
  Default is None or no initialization.
  It is recommended to use pre-trained model for initialization and the pre-trained model can be obtained e.g. from [here](https://github.com/zengqunzhao/EfficientFace) (EfficientFace_Trained_on_AffectNet7.pth.tar is recommended)

- **lr** : *float, default=0.04*\
  Specifies the learning rate of the optimizer.

- **lr_steps**: *list of ints, default=[40, 55, 65, 70, 200, 250]*\
  Specifies the epochs on which learning rate is reduced by the factor of 10.

- **momentum**: *float, default=0.9*\
  Specifies the momentum of SGD optimizer.

- **dampening**: *float, default=0.9*\
  Specifies the dampening factor coefficient.

- **weight_decay**: *float, default=1e-3*\
  Specifies the weight decay coefficient.  

- **iters**: *int, default=100*\
  Specifies the number of epochs used to train the classifier. 
 
- **batch-size**: int, default=8\
  Specifies the minibatch size.

- **n_workers**: *int, default=4*\
  Specifies number of threads to be used in data loading. 
  
- **device**: *str, default='cpu', {'cuda', 'cpu'}*\
  Specifies the computation device.  

#### `AudiovisualEmotionLearner.fit`
```python
AudiovisualEmotionLearner.fit(self, dataset, val_dataset, logging_path, silent, verbose, eval_mode, restore_best):
)
```

Method to train the audiovisual emotion recognition model.
After calling this method the model is trained for specified number of iterations and the last checkpoint is saved.
If the validation set is provided, the best checkpoint on validation set is saved in addition.

Returns a dictionary containing a list of cross entropy measures (dict keys: `"train_loss"`, `"val_loss"`) and a list of accuracy (dict key: `"train_acc"`, `"val_acc"`) during the entire optimization process.  

**Parameters**:

- **dataset**: *engine.datasets.DatasetIterator*\
    OpenDR dataset object that holds the training set.
    The `__get_item__` should return audio data of shape (C x N), video data of shape (C x N x H x W).

- **val_dataset**: *engine.datasets.DatasetIterator, default=None*\
    OpenDR dataset object that holds the validation set.
    The `__get_item__` should return audio data of shape (C x N), video data of shape (C x N x H x W).
    If `val_dataset` is not `None`, it is used to select the model's weights that produce the best validation accuracy and save these weights.

- **logging_path**: *str, default='logs/'*\
    Path for saving Tensorboard logs and model checkpoints.  

- **silent**: *bool, default=False*\
    If set to True, disables all printing, otherwise, the performance statistics are printed to STDOUT after every 10th epoch.  
  
- **verbose**: *bool, default=True*\
    If set to True, the performance statistics are printed after every epoch.
 
- **eval_mode**: *str, default='audiovisual', {'audiovisual', 'noisyaudio', 'noisyvideo', 'onlyaudio', 'onlyvideo'}*\
   Evaluation mode of the model.
   Check [here](https://arxiv.org/abs/2201.11095) for details.
- **restore_best**: *bool, default=False*
  If set to true, best weights on validation set are restored in the end of training if validation set is provided, otherwise best weights on train set.

**Returns**:

  - **performance**: *dict*  
    A dictionary that holds the lists of performance curves with the following keys: `"train_acc"`, `"train_loss"`, `"val_acc"`, `"val_loss"`.


#### `AudiovisualEmotionLearner.eval`   
```python
AudiovisualEmotionLearner.eval(self, dataset, silent, verbose, mode)
```

This method is used to evaluate the current audiovisual emotion recognition model given a dataset.  
 
**Parameters**:

- **dataset**: *engine.datasets.DatasetIterator*\
  OpenDR dataset object that holds the training set.
  The `__get_item__` should return audio data of shape (C x N), video data of shape (C x N x H x W).

- **silent**: *bool, default=False*\
  If set to True, disables all prints

- **verbose**: *bool, default=True*\
  If set to True, prints the accuracy and performance of the model

- **mode**: *str, default='audiovisual', {'audiovisual', 'noisyaudio', 'noisyvideo', 'onlyaudio', 'onlyvideo'}\
   Evaluation mode of the model. Check [here](https://arxiv.org/abs/2201.11095) for details. 
 
**Returns**:

- **performance**: *dict*  
  Dictionary that contains `"cross_entropy"` and `"acc"` as keys.  


#### `AudiovisualEmotionLearner.infer`  
```python
AudiovisualEmotionLearner.infer(self, audio, video)
```

This method is used to generate the emotion prediction given an audio and a video.  
Returns an instance of `engine.target.Category` representing the prediction.  

**Parameters**:

- **audio**: *engine.data.Timeseries*\
  Object of type `engine.data.Timeseries` that holds the input audio data. 

- **video**: *engine.data.Video*\
  Object of type `engine.data.Video` that holds the input video data.

**Returns**:

- **prediction**: *engine.target.Category*\
  Object of type `engine.target.Category` that contains the prediction.  


#### `AudiovisualEmotionLearner.save`  
```python
AudiovisualEmotionLearner.save(path, verbose)
```

This method is used to save the current model instance under a given path.  
The saved model can be loaded later by calling `AudiovisualEmotionLearner.load(path)`.   
Two files are saved under the given directory path, namely `"path/metadata.json"` and `"path/model_weights.pt"`.  
The former keeps the metadata and the latter keeps the model weights.   

**Parameters**:

- **path**: *str*\
  Directory path to save the model.   
- **verbose**: *bool, default=True*\
  If set to True, print acknowledge message when saving is successful.   
  
#### `AudiovisualEmotionLearner.download`  
```python
AudiovisualEmotionLearner.download(self, path)  
```

This method is used to download a pretrained model from a given directory.
Pretrained models are provided for the 'ia' fusion with 'zerodrop' modality dropout.
The pretrained model is trained on [RAVDESS](https://zenodo.org/record/1188976#.YlkXNyjP1PY) dataset which is under CC BY-NC-SA 4.0 license.

**Parameters**:

- **path**: *str*\
  Directory path to download the model.
  Under this path, "metadata.json" and "model_weights.pt" will be downloaded.
  The weights of the downloaded pretrained model can be loaded by calling the AudiovisualEmotionLearner.load method.

#### `AudiovisualEmotionLearner.load`  
```python
AudiovisualEmotionLearner.load(self, path, verbose)  
```

This method is used to load a previously saved model (by calling `AudiovisualEmotionLearner.save`) from a given directory.  
Note that under the given directory path, `"metadata.json"` and `"model_weights.pt"` must exist.   

**Parameters**:

- **path**: *str*\
  Directory path of the model to be loaded.  

- **verbose**: *bool, default=True*\
  If set to True, print acknowledge message when model loading is successful.  

#### `opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm.data.get_audiovisual_emotion_dataset`
```python
opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm.data.get_audiovisual_emotion_dataset(path, sr, n_mfcc, preprocess, target_time=, input_fps, save_frames, target_im_size, device
)
```

**Parameters**:
  
- **path**: *str*\
  Specifies the directory path where the dataset resides. 
  For training on [Ravdess dataset](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391) please download the data from [here](https://zenodo.org/record/1188976#.YlCAwijP2bi). 

- **sr**: *int, default=22050*\
  Specifies sampling rate for audio processing.

- **n-mfcc**: *int, default=10*\
  Specifies number of MFCC featuers to extract from audio.

- **preprocess**: *bool, default=False*\
  Preprocesses the downloaded Ravdess dataset for training and saves the processed data files.

- **target_time**: *float, default=3.6*\
  Target in seconds time to which the videos and audios are cropped/padded.

- **input_fps**: *int, default=30*\
  Frames per second of input video file

- **save_frames**: *int, defauult=15*\
  Number of frames from the video to keep for training

- **target_im_size**: *int, default=224*\
  Target height and width of video to which the video is reshaped

- **device**: *{'cpu', 'cuda'}*\
  Device on which to run the preprocessing (has no effect if preprocess=False)

**Returns**:

- **train_set**: *opendr.engine.datasets.DatasetIterator*\
  The training set object that can be used with the `fit()` method to train the hand gesture classifier.   

- **val_set**: *opendr.engine.datasets.DatasetIterator*\
  The validation set object that can be used with the `fit()` method to train the hand gesture classifier.   

- **test_set**: *opendr.engine.datasets.DatasetIterator*\   
  The test set object that can be used with the `fit()` method to train the hand gesture classifier.   


### Examples

* **Training an audio and vision emotion recognition model**.  
  In this example, we will train an audiovisual emotion recognition model on [Ravdess](https://zenodo.org/record/1188976#.YlCAwijP2bi) dataset.
  First, the dataset needs to be downloaded (the files Video_Speech_Actor_[01-24].zip and Audio_Speech_Actors_01-24.zip).
  The directory should be organized as follows:
    ```
    RAVDESS
    └───ACTOR01
    │   │  01-01-01-01-01-01-01.mp4
    │   │  01-01-01-01-01-02-01.mp4
    │   │  ...
    │   │  03-01-01-01-01-01-01.wav
    │   │  03-01-01-01-01-02-01.wav
    │   │  ...
    └───ACTOR02
    └───...
    └───ACTOR24
    ```
    The training, validation, and testing data objects can be constructed easily by using our method `get_audiovisual_emotion_dataset()` as follows: 
    
  ```python
  from opendr.perception.multimodal_human_centric import get_audiovisual_emotion_dataset
  from opendr.perception.multimodal_human_centric import AudiovisualEmotionLearner
  
  train_set, val_set, test_set = get_audiovisual_emotion_dataset('RAVDESS/', preprocess=True)
  ```
  When the data is downloaded and processed for the first time, it has to be preprocessed by setting the argument `preprocess=True` in `get_audiovisual_emotion_dataset()`.
  This will preprocess the audio and video files and save the preprocessed files as numpy arrays, as well as create a random train-val-test split.
  The preprocessing should be ran once and `preprocess=False` should be used in subsequent runs on the same dataset.
   
  Then, we can construct the emotion recognition model, and train the learner for 100 iterations as follows:  

  ```python
  learner = AudiovisualEmotionLearner(iters=100, device='cuda')
  performance = learner.fit(train_set, val_set, logging_path = 'logs', restore_best=True)
  learner.save('model')
  ```
  Here, we restore the best performance on the validation set and save the model.
  The training logs are saved under logs/.
  
  Additionally, pretrained EfficientFace model can be obtained from [here](https://github.com/zengqunzhao/EfficientFace) and used for weight initialization of the vision backbone by setting `pretr_ef='EfficientFace_Trained_on_AffectNet7.pth.tar'`.
  
* **Using pretrained audiovisual emotion recognition model**

  In this example, we will demonstrate how a pretrained audiovisual emotion recognition model can be used.
  First, we will download the pre-trained model.
  The model is trained on RAVDESS [3] dataset under CC BY-NC-SA 4.0 license.

   ```python
  from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.avlearner import AudiovisualEmotionLearner 
  
  learner = AudiovisualEmotionLearner()
  learner.download('model')
  learner.load('model')
  ```

  Given an input video and audio, we can preprocess the data and make inference using the pretrained model as follows:

  ```python
  audio, video = avlearner.load_inference_data(args.input_audio, args.input_video)
  prediction = avlearner.infer(audio, video)
  ```
    
#### References
[1] https://github.com/zengqunzhao/EfficientFace
[2] https://github.com/okankop/Efficient-3DCNNs
[3] https://zenodo.org/record/1188976#.YlCAwijP2bi
