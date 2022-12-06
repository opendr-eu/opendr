# SiamRPNLearner Module

This class implements the SiamRPN generic object tracker based on its
[GluonCV](https://github.com/dmlc/gluon-cv) implementation.

## Tracking datasets

### Training datasets

The following datasets are supported for training the SiamRPN tracker:

1. COCO Detection dataset ([preprocessing scripts](https://github.com/foolwood/SiamMask/tree/master/data/coco))
2. YouTube BB dataset ([preprocessing scripts](https://github.com/foolwood/SiamMask/tree/master/data/ytb_vos))
3. ILSVRC-VID ([preprocessing scripts](https://github.com/foolwood/SiamMask/tree/master/data/vid))
4. ILSVRC-DET ([preprocessing scripts](https://github.com/foolwood/SiamMask/tree/master/data/det))

The datasets need to be downloaded and preprocessed as indicated in the
[SiamMask](https://github.com/foolwood/SiamMask/tree/master/data) GitHub repository.

The following data structure is expected:

```
data_root
├── ...
├── coco                 
│   ├── crop511 
│   ├── ...
├── Youtube_bb                 
│   ├── crop511 
│   ├── ...
├── vid                 
│   ├── crop511 
│   ├── ...
├── det                 
│   ├── crop511 
│   ├── ...
└── ...
```

#### Custom training datasets

Support for custom datasets is implemented by inheriting the `opendr.engine.datasets.DatasetIterator` class as shown in
[otb_dataset.py](/src/opendr/perception/object_tracking_2d/datasets/otb_dataset.py). 

### Evaluation datasets

The [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html) dataset is supported
for tracker evaluation. The dataset can be downloaded using the [otb.py](data_utils/otb.py) script or the
`SiamRPNLearner.download('otb2015')` method as shown in
[eval_demo.py](/projects/python/perception/object_tracking_2d/demos/siamrpn/eval_demo.py).

The OpenDR SiamRPN model achieves a 66.8\% Success AUC on the OTB2015 dataset, running at ~132FPS
on an NVIDIA RTX 2070.
```shell
-------------------------------------------------------
|          Tracker name           | Success |   FPS   |
-------------------------------------------------------
|         siamrpn_opendr          |  0.668  |  132.2  |
-------------------------------------------------------
```