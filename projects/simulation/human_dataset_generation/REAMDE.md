# Human-Data-Generation-Framework

This folder contains the code for generating the data described in "Efficient Realistic Data Generation Framework leveraging Deep Learning-based Human Digitization"

## Download and reformat the CityScapes dataset

1. Download the CityScapes dataset from www.cityscapes-dataset.net <br />
    * RGB images: (a) leftImg8bit_trainvaltest.zip,  (b) leftImg8bit_trainextra.zip <br />
    * Annotation images: gtCoarse.zip <br />

The folder hierarchy should look like this:
```
├─ background_images
|  ├─ in
|  |  └─ CityScapes
|  |      ├─ leftImg8Bit
|  |      └─ gtCoarse
|  └─ out
|      
...
```
2. Run the following script to reformat the CityScapes dataset
```
python create_background_images.py
```
3. Run the following script to generate the dataset
```
python create_dataset.py
```   
## Citation
If you make use of the dataset, please cite the following reference in any publications:
```
@inproceedings{symeonidis2021data,
  title={Efficient Realistic Data Generation Framework leveraging Deep Learning-based Human Digitization},
  author={Symeonidis, C. and Nousi, P. and Tosidis, P. and Tsampazis, K. and Passalis, N. and Tefas, A. and Nikolaidis, N.}
  booktitle={Proceedings of the International Conference on Engineering Applications of Neural Networks (EANN)},
  year={2021}
}
```



# FaceRecognition Demos

This folder contains sample applications that demonstrate various parts of the functionality provided by the FaceRecognition algorithms provided by OpenDR.

More specifically, the following applications are provided:

1. demos/inference_tutorial.ipynb: A step-by-step tutorial on how to run inference using OpenDR's implementation of FaceRecognition
2. demos/eval_demo.py: A tool that demonstrates how to perform evaluation using FaceRecognition
3. demos/inference_demo.py: A tool that demonstrates how to perform inference on a single image
4. demos/benchmarking_demo.py: A simple benchmarking tool for measuring the performance of FaceRecognition in various platforms

Please use the --device cpu flag for the demos if you are running them on a machine without a CUDA-enabled GPU.
