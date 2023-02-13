# EfficientPSLearner Demo

The file [example_usage](./example_usage.py) contains code snippets explaining the following use cases:

1. Download the provided pre-trained model weights.

2. Prepare the supported datasets (Cityscapes, KITTI) after having downloaded the raw files. See the [datasets' readme](../../../../../src/opendr/perception/panoptic_segmentation/datasets/README.md) for further details.

3. Train the model.

4. Evaluate a trained model.

5. Run inference on RGB images.

**Note**: Please do not forget to adjust the `DATA_ROOT` variable to match your local paths.
