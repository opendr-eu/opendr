# Panoptic Segmentation

Panoptic segmentation combines both semantic segmentation and instance segmentation in a single task.
While distinct foreground objects, e.g., cars or pedestrians, receive instance-wise segmentation masks, background classes such as buildings or road surface are combined in class-wide labels. 

## Modules

### EfficientPS: Efficient Panoptic Segmentation

For the task of panoptic segmentation, EfficientPS has been included in the OpenDR toolkit.
The model architecture leverages a shared backbone for efficient encoding and fusing of semantically rich multi-scale features.
Two separate network heads create predictions for semantic and instance segmentation, respectively.
The final panoptic fusion model combines the output of the task-specific heads into a single panoptic segmentation map.

Website: http://panoptic.cs.uni-freiburg.de <br>
GitHub repository: https://github.com/DeepSceneSeg/EfficientPS

**BibTeX**:
```bibtex
@article{mohan2020efficientps,
  title={Efficientps: Efficient panoptic segmentation},
  author={Mohan, Rohit and Valada, Abhinav},
  journal={International Journal of Computer Vision (IJCV)},
  year={2021}
}
```

**Base repositories**

The OpenDR implementation extends the [EfficientPS repository](https://github.com/DeepSceneSeg/EfficientPS), from [Rohit Mohan](https://rl.uni-freiburg.de/people/mohan) and [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), with the OpenDR interface.
In order to comply with the OpenDR style references, minor changes have been applied such as formatting and fixing PEP8 issues.

Please note that the original repository is heavily based on
- [mmdetection](https://github.com/open-mmlab/mmdetection) by the [OpenMMLab](https://openmmlab.com/) project
- [gen-efficient-net-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch) authored by [Ross Wightman](https://github.com/rwightman)
