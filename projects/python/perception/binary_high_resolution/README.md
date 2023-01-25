# Binary High Resolution Analysis Demos

This folder contains minimal code usage examples that showcase the basic functionality of the BinaryHighResolutionLearner provided by OpenDR.
Specifically the following examples are provided:

1. **train_eval_demo.py**: Fit learner to a toy set.
2. **inference_demo.py**: Perform inference on a single image.
3. **onnx_inference_demo.py**: Optimize a model using ONNX and perform inference on a single image.

For the last two demos it is required to have a trained model.
Please find turn **train_eval_demo.py** in order to save a model before running these demos.

Example usage:
```shell
python3 train_eval_demo.py
```