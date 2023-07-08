## vosk module

The *vosk* module contains the *VoskLearner* class, which inherits from the abstract class *Learner*.

### Class VoskLearner

Bases: `engine.learners.Learner`

The *VoskLearner* class is a wrapper of libary [[1]](#alphacep/vosk-api/python-github) implementation. It is integrated for the speech transcription task.


The [VoskLearner](/src/opendr/perception/speech_transcription/vosk/vosk_learner.py) class has the following public methods:

#### `WhisperLearner` constructor

```python
VoskLearner(self, device, sample_rate)
```

Constructor parameters:

- **device**: *str, default="cpu"*\
    The device to use for computations. Currently only supports cpu.

- **sample_rate**: *int, default=16000*\
    The sample rate to be used by the Vosk model.

#### `VoskLearner.eval`

```python
VoskLearner.eval(self, dataset, save_path_csv)
```

This method is used to evaluate Vosk model on the given dataset.

Returns a dictionary containing evaluation metrics such as word error rate.

Parameters:

- **dataset**: *DatasetIterator*\
  A speech dataset.
- **save_path_csv**: *Optional[str], default=None*\
  The path to save the evaluation results.

#### `VoskLearner.infer`

```python
VoskLearner.infer(self, audio)
```

This method runs inference on an audio sample. Please call the load() method before calling this method.

Return transcription as `VoskTranscription` that contains transcription text and other side information.

Parameters:
- **audio**: *Union[Timeseries, torch.Tensor, np.ndarray, bytes]*\
    The audio sample as a `Timeseries`, `torch.Tensor`, or `np.ndarray` or `bytes`.

#### `VoskLearner.load`

```python
VoskLearner.load(self, name, language, model_path, download_dir)
```

This method loads the Vosk model and initializes the recognizer. The method will download model if necessary

Parameters:

- **name**: *Optional[str], default=None*\
    Full name of the Vosk model.

- **language**: *Optional[str], default=None*\
    Language of the Vosk model. Vosk will decide the default model for this language.

- **model_path**: *Optional[str], default=None*\
    Path to the Vosk model.

- **download_dir**: *Optional[bool], default=False*\
    Directory to download the Vosk model to.


#### `VoskLearner.download`

```python
VoskLearner.download(self, model_name)
```

Download model given a local path including the full name of the Vosk model.

Parameters:
- **model_name**: *Path*\
    Path to download model to, including the full name of the Vosk model.


#### Examples

* **Download and load a model by its name and infer a sample from an existing file.**
```python
import librosa
import numpy as np

from opendr.engine.data import Timeseries
from opendr.perception.speech_transcription import VoskLearner

learner = VoskLearner()
learner.load(language="en-us")

# Assuming you have recorded your own voice sample in command.wav in the current directory
signal, sampling_rate = librosa.load("video.wav", sr=learner.sample_rate)
signal = np.expand_dims(signal, axis=0)
timeseries = Timeseries(signal)
result = learner.infer(timeseries)
print(result)
```

#### References

<a name="alphacep/vosk-api/python-github" href="https://github.com/alphacep/vosk-api/tree/master/python">[1]</a>
Github: [alphacep/vosk-api](https://github.com/alphacep/vosk-api/tree/master/python).

