# Speech Command Recognition

This folder contains a demo script for performing the speech command recognition on a given audio file, using one of the four currently available models: MatchboxNet, EdgeSpeechNets, Quadratic SelfONN or Whisper. The script uses a pretrained model that can be either provided from a local file or downloaded from the sever hosted by OpenAI (Whisper) or OpenDR FTP server (the latter option is currently not available for EdgeSpeechNets).


## Demo with audio file

### Example:

1. Download and load model. 

1.1. Download and load model without specify location. The implementation will download to some default folders: Current directory for Whisper. To use the buitlin transcribe implementation of Whisper, set `buitlin_transcribe` to True in command line argument.
```
python demo.py example1.wav --backbone whisper --model-name tiny.en
```
```
python demo.py example1.wav --backbone vosk --lang en-us
```

1.2. Download model to a specified path and load it to memory.
```
python demo.py example1.wav --backbone whisper --model-name --download-dir "./whisper_model"
```
```
python demo.py example1.wav --backbone vosk --lang en-us --download-dir "./vosk_model"
```

1.3. Load model from a path. For Whisper, path is a file, and for Vosk, path is a directory.
```
python demo.py example1.wav --backbone whisper --model-name tiny.en --model-path "./whisper_model/tiny.en.pt"
```
```
python demo.py example1.wav --backbone vosk --language en-us --model-path "./vosk_model/vosk-model-small-en-us-0.15"
```


## Live Demo

The `demo_live.py` is a simple command line tool that continuously record and transcribe audio in a loop. It waits for the user to say "Hi Whisper" or "Hi Vosk" before starting the loop, and stops the loop when the user says "Bye Whisper" or "Bye Vosk". The tool can be configured using several command line arguments.

### Example

Here is an example command that records and transcribes audio every 0.5 seconds, with a recording duration of 2 seconds, using the default Whisper model:

```
python demo_live.py -d 5 -i 0.25 --backbone whisper --model-name tiny.en --language en --device cuda
```

```
python demo_live.py -d 5 -i 0.25 --backbone vosk --language en-us
```


