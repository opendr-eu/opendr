# KITTI Odometry Data
1. Download the KITTI Odometry data from http://www.cvlibs.net/datasets/kitti/eval_odometry.php:
- `odometry data set (color, 65 GB)`
- `odometry ground truth poses (4 MB)`

2. Download the KITTI raw data from http://www.cvlibs.net/datasets/kitti/raw_data.php for the runs specified in [`datasets/kitti.py`](datasets/kitti.py) (search for `KITTI_RAW_SEQ_MAPPING`). - `[synced+rectified data]`

3. Extract it into RAW_PATH folder and run the following line

```
python src/opendr/perception/continual_slam/datasets/kitti.py <RAW_PATH> <ODOMETRY_PATH> --oxts
```

