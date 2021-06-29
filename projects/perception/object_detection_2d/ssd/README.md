# SingleShotDetector Demos

This folder contains minimal code usage examples that showcase the basic functionality of the SingleShotDetectorLearner 
provided by OpenDR. Specifically the following examples are provided:
1. inference_demo.py: Perform inference on a single image. Setting `--device cpu` performs inference on CPU.
   
2. eval_demo.py: Perform evaluation on the `WiderPersonDataset`, implemented in OpenDR format. The user must first download 
   the dataset and provide the path to the dataset root via `--data-root /path/to/wider_person`. 
   Setting `--device cpu` performs evaluation on CPU. 
   
3. train_demo.py: Fit learner to dataset. PASCAL VOC and COCO datasets are supported via `ExternalDataset` class and any 
   `DetectionDataset` can be used as well. Provided is an example of training on `WiderPersonDataset`. The user must set the 
   dataset type using the `--dataset` argument and provide the dataset root path with the `--data-root` argument. 
   Setting `--device cpu` performs training on CPU. 
   
    Example usage:
   `python3 train_demo.py --dataset widerperson --data-root /path/to/wider_person`