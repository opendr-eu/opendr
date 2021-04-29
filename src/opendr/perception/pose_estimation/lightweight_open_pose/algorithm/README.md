# OpenDR Pose Estimation - Lightweight Open Pose

This folder contains the Open Pose[1] algorithm implementation for pose estimation in the OpenDR Toolkit, 
in the form of Lightweight Open Pose [2].

## Sources

The algorithms files are copied from [Daniil-Osokin/lightweight-human-pose-estimation.pytorch](
https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) with minor modifications listed below:

1. `datasets/coco.py`: PEP8 changes
2. `datasets/transformations.py`: PEP8 changes
3. `models/with_mobilenet`: PEP8 changes
4. `modules/conv.py`: PEP8 changes
5. `modules/get_parameters.py`: PEP8 changes
6. `modules/keypoints.py`: PEP8 changes
7. `modules/load_state.py`: 
    - Modified `load_state()` by adding a try/except block
    - Commented out warning prints from both functions
8. `modules/loss.py`: PEP8 changes
9. `one_euro_filter.py`:  PEP8 changes
10. `scripts/make_val_subset.py`: Modified to work as a callable function
11. `scripts/prepare_train_labels.py`:
    - PEP8 changes
    - Modified to work as a callable function
12. `val.py`: 
    - PEP8 changes
    - Removed `main`
    - Added verbose conditionals in `run_coco_eval()`

## Added Files
Two additional backbone models were added in the `models` directory.

- `models/with_mobilenet_v2.py`: Copied from [tonylins/pytorch-mobilenet-v2](
https://github.com/tonylins/pytorch-mobilenet-v2) and modified to work as a backbone for pose estimation.
- `models/with_shufflenet.py`: Copied from [jaxony/ShuffleNet](https://github.com/jaxony/ShuffleNet) and modified to 
work as a backbone for pose estimation. 

## References

- [1]: OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields, 
[arXiv](https://arxiv.org/abs/1812.08008).
- [2]: Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose, 
[arXiv](https://arxiv.org/abs/1811.12004).
