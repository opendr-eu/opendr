# Image-based Facial Expression Recognition Demo

This folder contains an implemented demo of image_based_facial_expression_recognition method implemented by [[1]](#1).
The demo framework has three main features:
- Image: recognizes facial expressions in images.
- Video: recognizes facial expressions in videos in a frame-based approach.
- Webcam: connects to a webcam and recognizes facial expressions of the closest face detected by a face detection algorithm.
The demo utilizes OpenCV face detector Haar Cascade [[2]](https://ieeexplore.ieee.org/abstract/document/990517) for real-time face detection.

#### Running demo
The pretrained models on AffectNet Categorical dataset are provided by [[1]](#1) which can be found [here](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/tree/master/model/ml/trained_models/esr_9).
**Please note that the pretrained weights cannot be used for commercial purposes**
To recognize a facial expression in images, run the following command:
```python
python inference_demo.py image -i ./media/jackie.jpg -d 
```  

The argument `image` indicates that the input is an image. The location of the image is specified after `-i` and `-d` sets the display mode to true.
If the location of image file is not specified, the demo automatically downloads a sample image file from the FTP server.

```python
python inference_demo.py image -i 'image_path' -d 
```  

To recognize a facial expression in videos, run the following command:
```python
python inference_demo.py video -i 'video_path' -d -f 5
```
The argument `video` indicates that the input is a video. The location of the video is specified after `-i`. `-d` sets the display mode to true, `-f` defines the number of frames to be processed.
If the location of video file is not specified, the demo automatically downloads a sample video file from the FTP server. 

To recognize a facial expression in images captured from a webcam, run the following command:
```python
python inference_demo.py webcam -d
```
The argument `webcam` indicates the framework to capture images from a webcam. `-d` sets the display mode to true.

#### List of Arguments
Positional arguments:

- **mode**:\
Select the running mode of the demo which are 'image', 'video' or 'webcam'.
Input values: {image, video, webcam}.

Optional arguments:

- **-h (--help)**:\
Display the help message.

- **-d (--display)**:\
Display a window with the input data on the left and the output data on the right (i.e., detected face, emotions, and affect values).

- **-i (--input)**:\
Define the full path to an image or video.

- **-c (--device)**:\
Specifies the device, which can be 'cuda' or 'cpu'.

- **-w (--webcam)**:\
Define the webcam to be used while the framework is running by 'id' when the webcam mode is selected. The default camera is used, if 'id' is not specified.

- **-f (--frames)**:\
Set the number of frames to be processed for each 30 frames. The lower is the number, the faster is the process.


## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.


## References
<a id="1">[1]</a>
[Siqueira, Henrique, Sven Magg, and Stefan Wermter. "Efficient facial feature learning with wide ensemble-based convolutional neural networks." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 04. 2020.](
https://ojs.aaai.org/index.php/AAAI/article/view/6037)

<a id="2">[2]</a>
[Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Proceedings of the 2001 IEEE computer society conference on computer vision and pattern recognition. CVPR 2001. Vol. 1. Ieee, 2001](
https://ieeexplore.ieee.org/abstract/document/990517)
