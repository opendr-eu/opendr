# Hand Gesture Recognition

This demo performs hand gesture recognition from webcam. The list of gestures is: 'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace inv', 'rock', 'stop', 'stop inv', 'three', 'three 2', 'two up', 'two up inv', 'no gesture', an example can be seen in [here](https://github.com/hukenovs/hagrid/tree/master). By default single person (two hands) is assumed, but this can be changed by setting `max_hands` parameter. The dataset is trained on images with distance to camera of 1..4 meters, so similar distance is expected to perform best.

Demo can be run as follows:
```python
python3 demo.py --max_hands 2 
```
