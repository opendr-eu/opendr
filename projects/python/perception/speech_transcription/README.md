# Speech Command Recognition

This folder contains a demo script for performing the speech command recognition on a given audio file, using one of the four currently available models: MatchboxNet, EdgeSpeechNets, Quadratic SelfONN or Whisper. The script uses a pretrained model that can be either provided from a local file or downloaded from the sever hosted by OpenAI (Whisper) or OpenDR FTP server (the latter option is currently not available for EdgeSpeechNets).


## Demo with audio file

The demo can be run with `demo.py` as follows:

```python
python demo.py INPUT_FILE --model [matchboxnet|edgespeechnets|quad_selfonn|whisper] 
```

The following additional parameters are supported:

` --model_path` gives the path to a pretrained model, if not given, downloading from the OpenDR FTP server will be attempted

` --n_class` defines the number of classes supported by the model (default 20). This parameters does not applied to Whisper.
 
The following argument only used for Whisper:

` --model_name` gives the name of different model of Whisper varying in number of parameters.


### Example:

1. Download and load model. 

1.1. Download and load model without specify location. The implementation will download to some default folders: Current directory for Whisper.
```
python demo.py example1.wav --model whisper --model-name tiny.en
```
```
python demo.py example1.wav --model vosk --lang en-us
```

1.2. Download model to a specified path and load it to memory.
```
python demo.py example1.wav --model whisper --model-name --download_dir "./pretrained_whisper"
```
```
python demo.py example1.wav --mode whisper --lang en-us --download_dir "./pretrained_vosk"
```

1.3. Load model from a path. For Whisper, path is a file, and for Vosk, path is a directory.
```
python demo.py example1.wav --model whisper --model-name tiny.en --model-path "./pretrained_whisper/tiny.en.pt"
```
```
python demo.py example1.wav --model vosk --language en-us --model-path "./pretrained_vosk/vosk-model-small-en-us-0.15"
```


## Live Demo

The `demo_live.py` is a simple command line tool that uses the `WhisperLearner` model to continuously record and transcribe audio in a loop. It waits for the user to say "Hi Whisper" before starting the loop, and stops the loop when the user says "Bye Whisper". The tool can be configured using several command line arguments.

Here is a list of available arguments:

- `-d` or `--duration`: Duration of the recording in seconds (default: 5)
- `-i` or `--interval`: Time interval between recordings in seconds (default: 10.0)
- `--device`: Device for running inference (default: "cpu")
- `-l` or `--load_path`: Path to the pretrained Whisper model. Use when you already download the model. 
- `-p` or `--download_path`: Save path for the download pretrained Whisper model (required if not using the default model)
- `--model_name`: Name of the pretrained Whisper model (default: "tiny.en")

### Example

Here is an example command that records and transcribes audio every 0.5 seconds, with a recording duration of 2 seconds, using the default Whisper model:

```
python demo_live.py -d 2 -i 0.5 --model_name tiny.en --device cuda
```

## Whisper Learner Speech Commands Benchmark

The `benchmark.py` evaluate the performance of the Whisper Learner on the Speech Commands dataset.

Here is a list of available arguments:

- `--root`: Root directory of the speech commands dataset (default: `./`)
- `--url`: URL of the speech commands dataset (choices: `speech_commands_v0.01`, `speech_commands_v0.02`; default: `speech_commands_v0.02`)
- `--folder_in_archive`: Folder name inside the archive of the speech commands dataset (default: `SpeechCommands`)
- `--subset`: Subset of the dataset to use (choices: `testing`, `validation`, `training`; default: `testing`)
- `--device`: Device to use for processing (`cpu` or `cuda`; default: `cuda`)
- `--batch_size`: Batch size for DataLoader (default: `8`)
- `--load_path`: Path to model checkpoint (required)
- `--model_name`: Whisper model name (default: `tiny.en`)
- `--fp16`: Inference with FP16 (default: `False`)

### Example

```
python benchmark.py --root ./ --model_name "tiny.en" --fp16 True --batch_size 8
```
