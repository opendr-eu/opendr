# Skeleton-based Human Action Recognition Demo

This folder contains an implemented demo of ensemble_based_facial_expression_recognition method implemented by [[1]](#1).
The demo framework has three main features:
- Image: recognizes facial expressions in images.
- Video: recognizes facial expressions in videos in a frame-based approach.
- Webcam: connects to a webcam and recognizes facial expressions of the closest face detected by a face detection algorithm.
The demo utilizes OpenCV face detector Haar Cascade [[2]](https://ieeexplore.ieee.org/abstract/document/990517) for real-time face detection. 

#### Data preparation
  Download the [AffectNet](http://mohammadmahoor.com/affectnet/) [[2]](https://www.computer.org/csdl/magazine/mu/2012/03/mmu2012030034/13rRUxjQyrW) dataset, and organize it in the following structure:
  ```
  AffectNet/    
      Training_Labeled/
          0/
          1/
          ...
          n/
      Training_Unlabeled/
          0/
          1/
          ...
          n/
      Validation/
          0/
          1/
          ...
          n/
  ```
  In order to do that, you need to run the following function:
  ```python
  from opendr.perception.facial_expression_recognition.ensemble_based_cnn.algorithm.utils import datasets
  datasets.pre_process_affect_net(base_path_to_images, base_path_to_annotations, base_destination_path, set_index)
  ```
  This pre-processes the AffectNet dataset by cropping and resizing the images into 96 x 96 pixels, and organizing them in folders with 500 images each.
  Each image is renamed to follow the pattern "[id][emotion_idx][valence times 1000]_[arousal times 1000].jpg".

#### Running demo
The pretrained models on AffectNet Categorical dataset are provided by [[1]](#1) which can be found [here](https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/tree/master/model/ml/trained_models/esr_9).
Please note that the pretrained weights cannot be used for commercial purposes!
To recognize a facial expression in images, run the following command:
```python
python inference_demo.py image -i ./media/jackie.jpg -d -s 2
```  

The argument "image" indicates that the input is an image. The location of the image is specified after -i while -d sets the display mode to true and -s 2 sets the window size to 1440 x 900.
You can also visualize regions in the image relevant for the classification of facial expression by adding -g as arguments:
```python
python inference_demo.py image -i 'image_path' -d -s 2 -g
```  
The argument -g generates saliency maps with the Grad-CAM algorithm.

To recognize a facial expression in videos, run the following command:
```python
python inference_demo.py video -i 'video_path' -d -f 5 -s 2
```
The argument "video" indicates that the input is a video. The location of the video is specified after -i. -d sets the display mode to true, -f defines the number of frames to be processed, and -s 2 sets the window size to 1440 x 900.

To recognize a facial expression in images captured from a webcam, run the following command:
```python
python inference_demo.py webcam -d -s 2 
```
The argument "webcam" indicates the framework to capture images from a webcam. -d sets the display mode to true, -s 2 sets the window size to 1440 x 900.

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

- **-s (--size)**:\
Define the size of the window:
1920 x 1080.
1440 x 900.
1024 x 768.
Input values: {1, 2, 3}.

- **-np (--no_plot)**:\
Hide the graph of activation and (un)pleasant values.

- **-c (--cuda)**:\
Run facial expression recognition on GPU.

- **-w (--webcam)**:\
Define the webcam to be used while the framework is running by 'id' when the webcam mode is selected. The default camera is used, if 'id' is not specified.

- **-f (--frames)**:\
Set the number of frames to be processed for each 30 frames. The lower is the number, the faster is the process.

- **-o (--output)**:\
Create and write ESR-9's outputs to a CSV file.
The file is saved in a folder defined by this argument (ex. '-o ./' saves the file with the same name as the input file in the working directory).

- **-g (--gradcam)**:\
Run the grad-CAM algorithm and shows the saliency maps with respect to each convolutional branch.


## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.


## References
<a id="1">[1]</a>
[Siqueira, Henrique, Sven Magg, and Stefan Wermter. "Efficient facial feature learning with wide ensemble-based convolutional neural networks." Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 04. 2020.](
https://ojs.aaai.org/index.php/AAAI/article/view/6037)

<a id="2">[2]</a>
[Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Proceedings of the 2001 IEEE computer society conference on computer vision and pattern recognition. CVPR 2001. Vol. 1. Ieee, 2001](
https://ieeexplore.ieee.org/abstract/document/990517)
