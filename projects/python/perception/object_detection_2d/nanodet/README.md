# NanoDet Demos

This folder contains minimal code usage examples that showcase the basic functionality of the NanodetLearner
provided by OpenDR. Specifically the following examples are provided:
1. inference_demo.py: Perform inference on a single image in a directory. Setting `--device cpu` performs inference on CPU.
   Setting the config file for the specific model is done with `--model "model name"`.
   Inference will use optimization [ONNX, JIT or TensorRT] if specified as `--optimize onnx`, `--optimize jit` or `--optimize trt`.
   If optimization is used, first an optimized model will be exported and then inference will be performed.

   In ONNX it is recommended to install `onnxsim` dependencies with `pip install onnxsim` on OpenDR's virtual environment, for smaller and better optimized models.

2. webcam_demo.py: A simple tool that performs live object detection using a webcam.

3. eval_demo.py: Perform evaluation on the `COCO dataset`, implemented in OpenDR format. The user must first download
   the dataset and provide the path to the dataset root via `--data-root /path/to/coco_dataset`.
   Setting `--device cpu` performs evaluation on CPU.

4. train_demo.py: Fit learner to dataset. PASCAL VOC and COCO datasets are supported via the `ExternalDataset` class.
   An example of training on the COCO dataset is provided. The user must set the dataset type using the `--dataset`
   argument and provide the dataset root path with the `--data-root` argument. Setting the config file for the specific
   model is done with `--model "model name"`. Setting `--device cpu` performs training on CPU. Additional command
   line arguments can be set to overwrite various training hyperparameters from the provided config file, run `python3 train_demo.py -h` to print information about them on stdout.

    Example usage:
   `python3 train_demo.py --model m --dataset coco --data-root /path/to/coco_dataset`

5. export_c_compatible_network.py: A simple example to export any model to be used with the C API of OpenDR.
   Note that this export will not be the same as the JIT optimization model used for inference in Python API, but it will perform the same.

6. inference_tutorial.ipynb: A simple tutorial in jupyter for using the Nanodet tool for inference.