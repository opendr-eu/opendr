# SiamRPNLearner Demos

This folder contains minimal code usage examples that showcase the basic functionality of the SiamRPNLearner 
provided by OpenDR. Specifically the following examples are provided:
1. inference_demo.py: Perform inference on a video. Setting `--device cpu` performs inference on CPU.
   
2. eval_demo.py: Perform evaluation on the OTB dataset, supported as `ExternalDataset` type. The toolkit
   provides the option to download the dataset at the location set by `--data-root /path/to/otb`. 
   Setting `--device cpu` performs evaluation on CPU. 
   
3. train_demo.py: Fit learner to dataset. COCO, ILSVRC-VID, ILSVRC-DET and Youtube-BB datasets are
   supported via `ExternalDataset` class.
   The user must set the dataset type using the `--datasets` argument and provide the datasets
   root path with the `--data-root` argument.
   See [here](/src/opendr/perception/object_tracking_2d/siamrpn/README.md) for the appropriate
   data folder structure.
   Setting `--device cpu` performs training on CPU. Additional command line arguments can be set
   to change various training hyperparameters, and running `python3 train_demo.py -h` prints
   information about them on stdout.
   
    Example usage:
   `python3 train_demo.py --dataset coco --data-root /path/to/coco2017`