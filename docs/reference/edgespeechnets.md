## edgespeechnets module

The *edgespeechnets* module contains the *EdgeSpeechNetsLearner* class, which inherits from the abstract class *Learner*
.

### Class EdgeSpeechNetsLearner

Bases: `engine.learners.Learner`

The *EdgeSpeechNetsLearner* class is a wrapper of the EdgeSpeechNets[[1]](#edgespeechnets-arxiv) implementation. It can
be used for limited-vocabulary speech command recognition tasks.

The [EdgeSpeechNetsLearner](#src.opendr.perception.speech_recognition.edgespeechnets_learner.py) class has the following public
methods:

#### `EdgeSpeechNetsLearner` constructor

```python
EdgeSpeechNetsLearner(self,
                      lr,
                      iters,
                      batch_size,
                      optimizer,
                      checkpoint_after_iter,
                      checkpoint_load_iter,
                      temp_path,
                      device,
                      architecture,
                      output_classes_n,
                      momentum,
                      preprocess_to_mfcc,
                      sample_rate
                      )
```

Constructor parameters:

- **lr**: *float, default=0.01*  
  Specifies the learning rate to be used during training.
- **epochs**: *int, default=30*  
  Specifies the number of epochs the training should run for.
- **batch_size**: *int, default=64*  
  Specifies number of images to be bundled up in a batch during training.
  This heavily affects memory usage, adjust according to your system.
  Should always be equal to or higher than the number of used CUDA devices.
- **optimizer**: *{'sgd'}, default='sgd'*  
  Specifies the optimizer to be used. Currently, only SGD is supported.
- **checkpoint_after_iter**: *int, default=0*  
  Specifies per how many training iterations a checkpoint should be saved. If set to 0 no checkpoints will be saved.
  Saves the models to the `temp_path` as "EdgeSpeechNet\<Architecture\>-\<epoch\>.pt"
- **checkpoint_load_iter**: *int, default=0*   
  Specifies a checkpoint to load based on the number of iterations before fitting.
  If set to 0 no checkpoint will be loaded.
- **temp_path**: *str, default='temp'*  
  Specifies the path to the directory where the checkpoints will be saved.
- **device**: *{'cpu', 'cuda'}, default='cuda'*  
  Specifies the device to be used.
- **architecture**: *{'A', 'B', 'C', 'D'}, default='A'*  
  Specifies the architecture to be used. See the [paper](#edgespeechnets-arxiv) for architecture model definitions.
- **output_classes_n**: *int, default=20*  
  Specifies the number of output classes the samples can be categorized to.
- **momentum**: *float, default=0.9*  
  Specifies the momentum for the SGD optimizer.
- **preprocess_to_mfcc**: *bool, default=True*  
  Specifies whether the learner should transform the input to a MFCC.
  If the input is already converted to a 2D signal,  turn this off.
  Expects a 1D signal if set to true.
- **sample_rate**: *int, default=16000*  
  Specifies the assumed sampling rate for the input signals used in the MFCC conversion.
  Does nothing if  *preprocess_to_mfcc* is set to false.

#### `EdgeSpeechNetsLearner.fit`

```python
EdgeSpeechNetsLearner.fit(self,
                          dataset,
                          val_dataset,
                          logging_path,
                          silent,
                          verbose)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Returns a dictionary containing stats regarding the last evaluation ran.
Parameters:

- **dataset**: *DatasetIterator*  
  Object that holds the training dataset. Will be converted used by a PyTorch Dataloader. Can be anything that can be
  passed to Dataloader as a dataset, but a safe way is to inherit it from DatasetIterator.
- **val_dataset**: *DatasetIterator, default=None*  
  Object that holds the validation dataset. Same rules apply as above.
- **logging_path**: *str, default=''*  
  Path to save TensorBoard log files. If set to None or '', TensorBoard logging is disabled.
- **silent**: *bool, default=True*  
  If set to True, disables all printing of training progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*  
  If set to True, enables additional log messages regarding model training.

#### `EdgeSpeechNetsLearner.eval`

```python
EdgeSpeechNetsLearner.eval(self, dataset)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:

- **dataset**: *DatasetIterator*  
  Object that holds the training dataset. Will be used by a PyTorch Dataloader.
  Can be anything that can be passed to Dataloader as a dataset, but a safe way is to inherit it from DatasetIterator.

#### `EdgeSpeechNetsLearner.infer`

```python
EdgeSpeechNetsLearner.infer(self, batch)
```

This method is used to classify signals. Can be used to infer a single utterance, or a batch of signals.

Parameters:

- **batch**: *Timeseries* or *List*[*Timeseries*]   
  Either a Timeseries or a list of Timeseries. 

#### `EdgeSpeechNetsLearner.save`

```python
EdgeSpeechNetsLearner.save(self, path)
```

This method saves the model in the directory provided by `path`.

Parameters:

- **path**: *str*  
  Path to the directory where the model should be saved. Does not need to exist before the function call.

#### `EdgeSpeechNetsLearner.load`

```python
EdgeSpeechNetsLearner.load(self, path)
```

This method loads the model from the directory provided by `path`.
In practice the same path as provided to `save` beforehand.

Parameters:

- **path**: *str*  
  Path to the model directory to be loaded.

#### Examples

* **Training example using randomized data in place of recorded samples.**

  ```python
  import numpy as np
  import os
  
  from OpenDR.engine.datasets import DatasetIterator
  from OpenDR.perception.speech_recognition.edgespeechnets.edgespeechnets_learner from EdgeSpeechNetsLearner
  
  class RandomDataset(DatasetIterator):
      def __init__(self):
          super().__init__()
          
      def __len__(self):
          return 64
  
      def __getitem__(self, item):
          return np.random.rand(16000), np.random.choice(10)
  
  learner = EdgeSpeechNetsLearner(output_clases_n=10, iters=10, architecture="A")
  training_dataset = RandomDataset()
  validation_dataset = RandomDataset()
  
  results = learner.fit(dataset=training_dataset, val_dataset=validation_dataset)
  # Print the validation accuracy of the last epoch and save the model to a file
  print(results[10]["validation_results"]["test_accuracy"])
  learner.save(os.path.join(".", "example", "directory", "path", "model"))
  ```

* **Load an existing model and infer a sample from an existing file.**
  ```python
  import librosa
  import numpy as np
  
  from OpenDR.engine.data import Timeseries
  from OpenDR.perception.speech_recognition.edgespeechnets.edgespeechnets_learner from EdgeSpeechNetsLearner
  
  learner = EdgeSpeechNetsLearner(output_clases_n=10, architecture="A")
  learner.load(os.path.join(".", "example", "directory", "path", "model"))

  signal, sampling_rate = librosa.load("command.wav", sr=learner.sample_rate)
  timeseries = Timeseries(signal)
  result = learner.infer(timeseries)
  print(result)
  ```

#### References

<a name="edgespeechnets-arxiv" href="https://arxiv.org/abs/1810.08559">[1]</a> EdgeSpeechNets: Highly Efficient Deep
Neural Networks for Speech Recognition on the Edge,
[arXiv](https://arxiv.org/abs/1810.08559).  
