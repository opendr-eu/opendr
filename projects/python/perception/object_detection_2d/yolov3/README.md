# YOLOv3DetectorLearner Demos

This folder contains minimal code usage examples that showcase the basic functionality of the YOLOv3DetectorLearner 
provided by OpenDR. Specifically the following examples are provided:
1. inference_demo.py: Perform inference on a single image. Setting `--device cpu` performs inference on CPU.

2. webcam_demo.py: A simple tool that performs live object detection using a webcam.
   
3. eval_demo.py: Perform evaluation on the PASCAL VOC dataset, supported as `ExternalDataset` type. The user must first 
   download the dataset and provide the path to the dataset root via `--data-root /path/to/voc`. 
   Setting `--device cpu` performs evaluation on CPU. 
   
4. train_demo.py: Fit learner to dataset. PASCAL VOC and COCO datasets are supported via `ExternalDataset` class and any 
   `DetectionDataset` can be used as well. Provided is an example of training on `WiderPersonDataset`. The user must set the 
   dataset type using the `--dataset` argument and provide the dataset root path with the `--data-root` argument. 
   Setting `--device cpu` performs training on CPU. Additional command line arguments can be set to change various training 
   hyperparameters, and running `python3 train_demo.py -h` prints information about them on stdout.
   
    Example usage:
   `python3 train_demo.py --dataset widerperson --data-root /path/to/wider_person`