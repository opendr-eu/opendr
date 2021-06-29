# RetinaFace Demos

This folder contains minimal code usage examples that showcase the basic functionality of the RetinaFaceLearner provided by 
OpenDR. Specifically the following examples are provided:
1. inference_demo.py: Perform inference on a single image. Use argument `--backbone mnet` to run a model trained to detect masked 
   faces. Setting `--device cpu` performs inference on CPU.
   
2. eval_demo.py: Perform evaluation on the `WiderFaceDataset`, implemented in OpenDR format. The user must first download 
   the dataset and provide the path to the dataset root via `--data-root /path/to/wider_face`. Optionally, the user can 
   enable image pyramid and flipping during evaluation to increase performance, by providing the `--pyramid` and `--flip` 
   flags. Use argument `--backbone mnet` to run a model trained to detect masked faces. 
   Setting `--device cpu` performs evaluation on CPU. By default, a subset of 100 images is evaluated.
   
3. train_demo.py: Fit learner to dataset. Only `WiderFaceDataset` is supported currently. Additional landmark annotations 
   provided by the original implementation are downloaded automatically from the OpenDR FTP server during training if not 
   present in the provided `--data-root`. Setting `--device cpu` performs training on CPU.