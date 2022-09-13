# NanoDet Demos

This folder contains minimal code usage examples that showcase the basic functionality of the NanodetLearner 
provided by OpenDR. Specifically the following examples are provided:
1. inference_demo.py: Perform inference on a single image in a directory. Setting `--device cpu` performs inference on CPU.
2. eval_demo.py: Perform evaluation on the `COCO dataset`, implemented in OpenDR format. The user must first download 
   the dataset and provide the path to the dataset root via `--data-root /path/to/coco_dataset`. 
   Setting `--device cpu` performs evaluation on CPU. 
   
3. train_demo.py: Fit learner to dataset. PASCAL VOC and COCO datasets are supported via `ExternalDataset` class.
   Provided is an example of training on `COCO dataset`. The user must set the dataset type using the `--dataset`
   argument and provide the dataset root path with the `--data-root` argument. Setting the config file for the specific
   model is done with `--model "wanted model name"`. Setting `--device cpu` performs training on CPU. Additional command
   line arguments can be set to overwrite various training hyperparameters from the provided config file, and running 
   `python3 train_demo.py -h` prints information about them on stdout.
   
    Example usage:
   `python3 train_demo.py --model plus-m_416 --dataset coco --data-root /path/to/coco_dataset`