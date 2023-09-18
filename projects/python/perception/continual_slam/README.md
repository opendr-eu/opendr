# Continual SLAM Demo

The file [example_usage](./example_usage.py) contains code snippets explaining the following use cases:

1. Download the provided pre-trained model weights.

2. Prepare the supported datasets (SemanticKITTI) after having downloaded the raw files. See the [datasets' readme](../../../../../src/opendr/perception/continual_slam/datasets/README.md) for further details.

3. Train the model.

4. Run inference

To increase the usability, we have added download method for both pre-trained model weights and test data which we provide on our server. For reference, we are publishing depth maps for respective frames; however, CL-SLAM is mainly written for ROS usage (2 headed nodes), and for more reference please have a look at our [ROS](/projects/opendr_ws/README.md) and [ROS2](/projects/opendr_ws_2/README.md) nodes. You can find more information about CL-SLAM usage in there nodes.