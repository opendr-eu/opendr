# DETR Demos

This folder contains minimal code usage examples that showcase the basic functionality of the DetrLearner provided by OpenDR. 
Specifically the following examples are provided:
1. **inference_demo.py**:   
   Perform inference on an RGB and an infrared image.
   Setting `--device cpu` performs inference on CPU.
   Setting `--backbone resnet50` performs inference with a resnet50 backbone.
   Setting `--panoptic-segmentation` performs panoptic segmentation.
2. **eval_demo.py**:  
   Perform evaluation on the COCO dataset, supported as `ExternalDataset` type.
   The user must first download the dataset and provide the path to the dataset root via `--data-root /path/to/coco`. 
   Setting `--device cpu` performs evaluation on CPU.  
3. **train_demo.py**:   
   Fit learner to dataset.
   Provided is an example of training on `WiderPersonDataset`. 
   The user must set the dataset root path with the `--data-root` argument. 
   Setting `--device cpu` performs training on CPU.
   Additional command line arguments can be set to change various training hyperparameters, and running
   `python3 train_demo.py -h` prints information about them on stdout.
   
    Example usage:
   `python3 train_demo.py --data-root /path/to/coco`