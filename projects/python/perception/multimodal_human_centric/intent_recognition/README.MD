# Intent Recognition

We provide a demo for text-based intent recognition in demo_text.py as well as intent recognition based on transcribed speech in demo_speech.py where transciption is based on OpenDR implementation of Whisper. 

In both cases, the model is trained in a multi-modal manner based on visual, audio, and text data as described [here](TODO), but only a single modality (language) is used at inference time for speed considerations.

The provided pretrained models are pretrained on [MIntRec dataset](https://github.com/thuiar/MIntRec) and the set of intent classes is: [Complain, Praise, Apologise, Thank, Criticize, Agree, Taunt, Flaunt, Joke, Oppose, Comfort, Care, Inform, Advise, Arrange, Introduce, Leave, Prevent, Greet, Ask for help]. Prediction is made for each separate sentence. 



Demo can be run as follows:
```python
python3 demo_text.py --text_backbone TEXT_BACKBONE --cache_path CACHE_PATH --model_dir MODEL_DIR
```

where TEXT_BACKBONE is one of ['bert-base-uncased', 'albert-base-v2', 'bert-small', 'bert-mini', 'bert-tiny'], CACHE_PATH is the path where cache whiles will be saved, and MODEL_DIR is the path where pretrained model weights are located (or will be downloaded if they don't exist).

```python
python3 demo_speech.py --text_backbone TEXT_BACKBONE --cache_path CACHE_PATH --backbone whisper --model-name MODEL_NAME --download_dir DOWNLOAD_DIR
```

where TEXT_BACKBONE and CACHE_PATH are as above, MODEL_NAME corresponds to whisper model name ([tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large]), and DOWNLOAD_DIR corresponds to the path where both whisper and intent recognition model weights are saved.
