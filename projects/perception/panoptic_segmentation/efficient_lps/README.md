# EfficientLPSLearner Demo

The file [example_usage](./example_usage.py) contains code snippets explaining the following use cases:

1. Download the provided pre-trained model weights.

[//]: # (TODO: Verify)
2. Prepare the supported datasets (KITTI, Nuscenes) after having downloaded the raw files.
   See the [datasets' readme](../../../../src/opendr/perception/panoptic_segmentation/datasets/README.md)
   for further details.

3. Train the model.

4. Evaluate a trained model.

5. Run inference on LiDAR Data.

## Usage
```
    example_usage.py
        [--kitti_dir|-k <kitti_path>]
        [--cp_dir|-c <checkpoint_path>]
        [--log_dir|-l <log_path>]
        [--skip_download]
        [--skip_train]
        [--skip_eval]
        [--skip_infer]
        [--display_figure|-v]
        [--save_figure|-s]
        [--projected|-p]
        [--detailed|-d]
```

### Arguments:
#### Paths
- `--kitti_dir|-k <kitti_path>`: (Optional, default: `~/dat/kitti/dataset/`)
  Path to the root directory of the KITTI dataset.
  It must be the parent directory of the `sequences/` folder.
  
- `--cp_dir|-c <checkpoint_path>`: (Optional, default: `~/dat/cp/`)
  Path to the directory where the model checkpoints are to be saved to and loaded from.
  
- `--log_dir|-l <log_path>`: (Optional, default: `~/dat/log/`)
  Path to the directory where the training logs are to be saved.
  
#### Action Flags
- `--skip_download`: Do not perform the model download test.
  
- `--skip_train`: Do not perform the model training test.
  
- `--skip_eval`: Do not perform the model evaluation test.
  
- `--skip_infer`: Do not perform the model inference test.

#### Inference Output Flags

- `--display_figure|-v`: Display the resulting figure during runtime.
  
- `--save_figure|-s`: Save the resulting figure to disk as an image.
  
- `--projected|-p`: Show the resulting segmentation as a 2D image of the spherical projection of the 3D points.
If not set, then the results will be represented as an orthographic projection of the 3D point cloud.
  
- `--detailed|-d`: If `projected` display the results as a grid of the original input (ranges), the panoptic mask,
the semantic labels and the contours. Otherwise, display the panoptic labels blended with the input in a single image.
  Not used for unprojected representation.
