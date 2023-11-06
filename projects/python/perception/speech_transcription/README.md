# Speech Transcription

This folder contains a demo script for performing the speech transcription on a given audio file, using one of the two currently available library: Whisper or Vosk. The script uses a pretrained model that can be either provided from a local file or download online.


## Demo with audio file

### Example:

1. Download and load model without specify location.
```
python demo.py example1.wav --backbone whisper --model_name tiny.en
```
```
python demo.py example1.wav --backbone vosk --lang en-us
```

2. Download model to a specified path and load it to memory.
```
python demo.py example1.wav --backbone whisper --model_name --download_dir "./whisper_model"
```
```
python demo.py example1.wav --backbone vosk --lang en-us --download_dir "./vosk_model"
```

3. Load model from a path. For Whisper, path is a file, and for Vosk, path is a directory.
```
python demo.py example1.wav --backbone whisper --model_name tiny.en --model_path "./whisper_model/tiny.en.pt"
```
```
python demo.py example1.wav --backbone vosk --language en-us --model_path "./vosk_model/vosk-model-small-en-us-0.15"
```


## Live Demo

The `demo_live.py` is a simple command line tool that continuously record and transcribe audio in a loop. It waits for the user to say "start" before starting the loop, and stops the loop when the user says "stop".

```
python demo_live.py -d 5 -i 0.25 --backbone whisper --model_name tiny.en --language en --device cuda
```
```
python demo_live.py -d 5 -i 0.25 --backbone whisper --model_name tiny.en --language en --initial_prompt "Vosk" --device cuda
```
```
python demo_live.py -d 5 -i 0.25 --backbone vosk --language en-us
```

## Evaluate on a dataset
The script `eval.py` will evaluate Whipper `tiny.en` model and Vosk `vosk-model-small-en-us-0.15` model on the test-clean split of LibriSpeech dataset. The word error rate is reported.

```
pythonn eval.py
```