## intent_recognition module

The *intent_recognition* module contains the *IntentRecognitionModule* class and can be used to recognize 20 intents of a person based on text.
It is recommended to use *IntentRecognitionModule* together with *SpeechTranscriptionModule* to enable intent recognition based on transcribed speech.
The module supports multimodal training on face (vision), speech (audio), and text data to facilitate improved unimodal inference on text modality.

We provide data processing scripts and pre-trained model for [MIntRec dataset](https://github.com/thuiar/MIntRec).

### Class IntentRecognitionLearner

The learner has the following public methods:

#### `IntentRecognitionLearner` constructor
```python
IntentRecognitionLearner(self, text_backbone, mode, log_path, cache_path, results_path, output_path, device, benchmark)
```

Constructor parameters:

- **text_backbone**: *{"bert-base-uncased", "albert-base-v2", "prajjwal1/bert-small", "prajjwal1/bert-mini", "prajjwal1/bert-tiny"}, default="bert-base-uncased"*\
  Specifies the text backbone to be used. The name matches the corresponding huggingface hub model, e.g., [prajjwal1/bert-small](https://huggingface.co/prajjwal1/bert-small).
- **mode**: *{'language', 'joint'}, default="joint"*\
  Specifies the modality of the model. 'Language' corresponds to text-only model, 'Joint' corresponds to multimodal model with vision, audio, and language modalities trained jointly.
- **log_path**: *str, default="logs"*\
  Specifies the path where to store the logs.
- **cache_path**: *str, default="cache"*\
  Specifies the path for cache, mainly used for tokenizer files.
- **results_path**: *str, default="results"*\
  Specifies where to store the results (performance metrics).
- **output_path**: *str, default="outputs"*\
  Specifies where to store the outputs: trained models, predictions, etc.
- **device**: *str, default="cuda"*\
  Specifies the device to be used for training.
- **benchmark**: *{"MIntRec"}, default="MIntRec"*\
  Specifies the benchmark (dataset) to be used for training. The benchmark defines the class labels, feature dimensionalities, etc.

#### `IntentRecognitionLearner.fit`
```python
IntentRecognitionLearner.fit(self, dataset, val_dataset, verbose, silent)
```

This method is used for training the algorithm on a training dataset and validating on a validation dataset.

Parameters:

- **dataset**: *object*\
  Object that holds the training dataset.
- **val_dataset** : *object, default=None*\
  Object that holds the validation dataset.
- **verbose** : *bool, default=False*\
  Enables verbosity.
- **silent** : *bool, default=False*\
  Enables training in the silent mode, i.e., only critical output is produced.

#### `IntentRecognitionLearner.eval`
```python
IntentRecognitionLearner.eval(self, dataset, modality, verbose, silent, restore_best_model)
```

This method is used to evaluate a trained model on an evaluation dataset.

Parameters:

- **dataset** : *object*\
  Object that holds the evaluation dataset.
- **modality**: *str*, {'audio', 'video', 'language', 'joint'}\
  Specifies the modality to be used for inference. Should either match the current training mode of the learner, or for a learner trained in joint (multimodal) mode, any modality can be used for inference, although we do not recommend using only video or only audio.
- **verbose**: *bool, default=False*\
  If True, provides detailed logs.
- **silent**: *bool, default=False*\
  If True, run in silent mode, i.e., with only critical output.
- **restore_best_model** : *bool, default=False*\
  If True, best model according to performance on validation set will be loaded from self.output_path. If False, current model state will be evaluated.

#### `IntentRecognitionLearner.infer`
```python
IntentRecognitionLearner.infer(self, batch, modality)
```

This method is used to perform inference from given language sequence (text).
Returns a list of `engine.target.Category` objects, which contains calss predictions and confidence scores for each sentence in the input sequence.

Parameters:
- **batch**: *dict*\
  Dictionary with input data with keys corresponding to modalities, e.g. {'text': 'Hello'}.
- **modality**: *str, default='language'*\
  Modality to be used for inference. Currently, inference from raw data is only supported for language modality (text).

#### `IntentRecognitionLearner.save`
```python
IntentRecognitionLearner.save(self, path)
```
This method is used to save a trained model.

Parameters:

- **path**: *str*\
  Path to save the model.

#### `IntentRecognitionLearner.load`
```python
IntentRecognitionLearner.load(self, path)
```

This method is used to load a previously saved model.

Parameters:

- **path**: *str*\
  Path of the model to be loaded.

#### `IntentRecognitionLearner.download`
```python
IntentRecognitionLearner.download(self, path)
```

Downloads the provided pretrained model into 'path'.

Parameters:

- **path**: *str*\
  Specifies the folder where data will be downloaded. 

#### `IntentRecognitionLearner.trim`
```python
IntentRecognitionLearner.trim(self, modality)
```

This method is used to convert a model trained in a multimodal manner ('joint' mode) for unimodal inference. This will drop unnecessary layers corresponding to other modalities for computational efficiency.

Parameters:
- **modality**: *str, default='language'*\
  Modality to which to convert the model

#### Examples

Additional configuration parameters/hyperparameters can be specified in *intent_recognition_learner/algorithm/configs/mult_bert.py*.

* **Training, evaluation and inference example**

  ```python
  from opendr.perception.multimodal_human_centric import IntentRecognitionLearner
  from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.data.mm_pre import MIntRecDataset

  if __name__ == '__main__':
    # Initialize the multimodal learner
    learner = IntentRecognitionLearner(text_backbone='bert-base-uncased', mode='joint', log_path='logs', cache_path='cache', results_path='results', output_path='outputs')

    # Initialize datasets
    train_dataset = MIntRecDataset(data_path='/path/to/data/', video_data_path='/path/to/video', audio_data_path='/path/to/audio', text_backbone='bert-base-uncased', split='train')
    val_dataset = MIntRecDataset(data_path='/path/to/data/', video_data_path='/path/to/video', audio_data_path='/path/to/audio', text_backbone='bert-base-uncased', split='dev')
    test_dataset = MIntRecDataset(data_path='/path/to/data/', video_data_path='/path/to/video', audio_data_path='/path/to/audio', text_backbone='bert-base-uncased', split='test')

    # Train the model
    learner.fit(dataset, val_dataset, silent=False, verbose=True)
 
    # Evaluate the best according to validation set model on multimodal input
    out = learner.eval(test_dataset, 'joint', restore_best_model=True)

    # Evaluate the best according to validation set model on text-only input
    out_l = learner.eval(test_dataset, 'language', restore_best_model=True)

    # Keep only the text-specific layers of the model and drop the rest
    learner.trim('language')

    # Evaluate the trimmed model. Should produce the same result as out_l.
    out_l_2 = learner.eval(test_dataset, 'language', restore_best_model=False)
  ```
