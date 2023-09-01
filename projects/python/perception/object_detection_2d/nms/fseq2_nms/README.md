# FSeq2-NMS Demos

This folder contains minimal code usage examples that showcase the basic functionality of the FSeq2-NMS implementation 
provided by OpenDR. Specifically the following examples are provided:

1. inference_demo.py: Perform inference on a single image. Setting `--device cpu` performs inference on CPU.

2. eval_demo.py: Perform evaluation on the `PETS` dataset, implemented in OpenDR format.
   Setting `--device cpu` performs evaluation on CPU. 
   
3. train_demo.py: Fit learner to dataset. `PETS` dataset is supported.
   Setting `--device cpu` performs training on CPU. Additional command line arguments can be set to change various training 
   hyperparameters, and running `python3 train_demo.py -h` prints information about them on stdout.
   
