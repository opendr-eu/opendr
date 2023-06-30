# Intent Recognition

We provide a demo for text-based intent recognition in demo_text.py as well as intent recognition based on transcribed speech in demo_speech.py where transciption is based on openDR implementation of whisper. 

In both cases, the model is trained in a multi-modal manner based on visual, audio, and text data as described [here](TODO), but only a single modality is used at inference time for speed considerations.

The provided pretrained models are pretrained on [MIntRec dataset](https://github.com/thuiar/MIntRec) and the set of intent classes is: [Complain, Praise, Apologise, Thank, Criticize, Agree, Taunt, Flaunt, Joke, Oppose, Comfort, Care, Inform, Advise, Arrange, Introduce, Leave, Prevent, Greet, Ask for help].



Demo can be run as follows:
```python
python3 demo_text.py --text_backbone TEXT_BACKBONE --cache_path CACHE_PATH
```

```python
python3 demo_speech.py --text_backbone TEXT_BACKBONE --cache_path CACHE_PATH --backbone whisper --model-name MODEL_NAME --download_dir DOWNLOAD_DIR
```
