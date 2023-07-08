## whisper module

The *whisper* module contains the *WhisperLearner* class, which inherits from the abstract class *Learner*.

### Class WhisperLearner

Bases: `engine.learners.Learner`

The *WhisperLearner* class is a wrapper of Whisper libary [[1]](#openai/whisper-github) implementation. It is integrated for the speech transcription task.


The [WhisperLearner](/src/opendr/perception/speech_transcription/whisper/whisper_learner.py) class has the following public methods:

#### `WhisperLearner` constructor

```python
WhisperLearner(self, verbose, temperature, compression_ratio_threshold,logprob_threshold, 
               no_speech_threshold, condition_on_previous_tex, word_timestamps, prepend_punctuations,
               append_punctuations, language, sample_len, best_of, beam_size, patience, length_penalty,
               prompt, prefix, suppress_tokens, suppress_blank, without_timestamps, max_initial_timestamp, fp16, device)
```

Constructor parameters:

- **verbose**: *bool*\
  Whether to display the text being decoded to the console. If True, displays all the details.
  If False, displays minimal details. If None, does not display anything.
- **temperature**: *Union[float, Tuple[float, ...]], default=0.0*\
  Temperature for sampling. It can be a tuple of temperatures, which will be successively used
  upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.
- **compression_ratio_threshold**: *float, default=2.4*\
  If the gzip compression ratio is above this value, treat as failed.
- **logprob_threshold**: *float, default=-0.8*\
  If the average log probability over sampled tokens is below this value, treat as failed
- **no_speech_threshold**: *float, default=0.6*\
  If the no_speech probability is higher than this value AND the average log probability over sampled tokens is below `logprob_threshold`, consider the segment as silent.
- **condition_on_previous_text**: *bool, default=False*\
  If True, the previous output of the model is provided as a prompt for the next window;
  disabling may make the text inconsistent across windows, but the model becomes less prone to
  getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
- **word_timestamps**: *bool, default=False*\
  Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
  and include the timestamps for each word in each segment.
- **prepend_punctuations**: *str, default='\"'“¿([{-'*\
  If word_timestamps is True, merge these punctuation symbols with the next word.
- **append_punctuations**: *float, default='\"'.。,，!！?？:：”)]}、'*\
  If word_timestamps is True, merge these punctuation symbols with the previous word.
- **language**: *Optional[str], default='en'*\
  Language spoken in the audio, specify None to perform language detection.
- **beam_size**: *Optional[int], default=None*\
  Number of beams in beam search, only applicable when temperature is zero.
- **patience**: *Optional[float], default=None*\
  Optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search.
- **length_penalty**: *Optional[float], default=None*\
  Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default.
- **prompt**: *Optional[Union[str, List[int]]], default=None*\
  Text or tokens to feed as the prompt; for more info: https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
- **prefix**: *Optional[Union[str, List[int]]], default=None*\
  Text or tokens to feed as the prefix; for more info: https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
- **suppress_tokens**: *Optional[Union[str, Iterable[int]]], default=-1*\
  Comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations.
- **suppress_blank**: *bool, default=True*\
  Suppress blank outputs.
- **without_timestamps**: *bool, default=False*\
  Use <|notimestamps|> to sample text tokens only, the timestamp will be multiple of 30 seconds if the audio file is longer than 30 seconds.
- **max_initial_timestamp**: *Optional[float], default=1*\
  Limit the range of timestamp tokens that can be generated at the beginning of a sequence.
- **fp16**: *bool, default=True*\
  Whether to perform inference in fp16. fp16 is not available on CPU.
- **device**: *str, default="cuda"*\
  Device to use for PyTorch inference, either "cpu" or "cuda".

#### `WhisperLearner.eval`

```python
WhisperLearner.eval(self, dataset, save_path_csv)
```

This method is used to evaluate Whisper model on the given dataset.

Returns a dictionary containing evaluation metrics such as word error rate.

Parameters:

- **dataset**: *DatasetIterator*\
  A speech dataset.
- **save_path_csv**: *Optional[str], default=None*\
  The path to save the evaluation results.

#### `WhisperLearner.infer`

```python
WhisperLearner.infer(self, audio)
```

This method runs inference on an audio sample. Please call the load() method before calling this method.

Return transcription as `WhisperTranscription` that contains transcription text
and other side information.

Parameters:
- **audio**: *Union[Timeseries, np.ndarray, torch.Tensor, str]*\
  The audio sample as a `Timeseries`, `torch.Tensor`, or `np.ndarray` or a file path as `str`.


#### `WhisperLearner.load`

```python
WhisperLearner.load(self, name, model_path, download_dir, in_memory)
```

This method loads Whisper model using Whisper builtin load() method. This method
will download model if necessary.


Parameters:

- **name**: *Optional[str], default=None*\
  Name of Whisper model. Could be: tiny.en, tiny, base, base.en, etc.

- **model_path**: *Optional[str], default=None*\
  Path to model checkpoint.

- **download_dir**: *Optional[str], default=None*\
  Directory to save the downloaded model.

- **in_memory**: *Optional[bool], default=False*\
  Whether to load the model in memory.

#### `WhisperLearner.download`

```python
Whisper.download(self, name, download_dir)
```

This method downloads Whisper model.

Parameters:
- **name**: *Optional[str]*\
  Name or path of model.
- **download_dir**: *Optional[str], default=None*\
  Directory to save the downloaded model.

#### Examples

* **Download and load a model by its name and infer a sample from an existing file.**
```python
import librosa
import numpy as np

from opendr.engine.data import Timeseries
from opendr.perception.speech_transcription import WhisperLearner

learner = WhisperLearner(language="en")
learner.load(name="tiny.en")

# Assuming you have recorded your own voice sample in video.wav in the current directory
signal, sampling_rate = librosa.load("video.wav", sr=learner.sample_rate)
signal = np.expand_dims(signal, axis=0)
timeseries = Timeseries(signal)
result = learner.infer(timeseries)
print(result)
```

#### References

<a name="openai/whisper-github" href="https://github.com/openai/whisper">[1]</a>
Github: [openai/whisper](https://github.com/openai/whisper).
