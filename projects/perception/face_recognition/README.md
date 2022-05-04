# FaceRecognition Demos

This folder contains sample applications that demonstrate various parts of the functionality provided by the FaceRecognition algorithms provided by OpenDR.

More specifically, the following applications are provided:

1. demos/inference_tutorial.ipynb: A step-by-step tutorial on how to run inference using OpenDR's implementation of FaceRecognition
2. demos/eval_demo.py: A tool that demonstrates how to perform evaluation using FaceRecognition
3. demos/inference_demo.py: A tool that demonstrates how to perform inference on a single image
4. demos/benchmarking_demo.py: A simple benchmarking tool for measuring the performance of FaceRecognition in various platforms
5. demos/webcam_demo.py: A tool that demonstrates how to perform face detection and recognition with the use of a webcam.
   1. To use this tool you have to first create a database containing the faces to be recognised. To do this, you will have to prepare the face images using the [align](https://github.com/opendr-eu/opendr/blob/master/docs/reference/face-recognition.md#facerecognitionlearneralign) method of the tool and place them in a folder named `'cropped_images_path'` inside the `'demos'` directory.

Please use the --device cpu flag for the demos if you are running them on a machine without a CUDA-enabled GPU.