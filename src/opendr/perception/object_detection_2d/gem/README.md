
# OpenDR Multimodal Object Detection 2D - GEM

This folder contains an implementation of GEM Multimodal Object Detector in an OpenDR Learner class for 2D object detection. The input of the model is a list of two images, one from each modality e.g., RGB and Infrared, and the output consists of bounding boxes with class names and confidences.

## Sources

This module currently depends on OpenDR's DETR module which is a wrapper of [facebook/detr](https://github.com/facebookresearch/detr) with modifications to make them compatible with OpenDR specifications. The GEM paper can be found here: [GEM: Glare or Gloom, I Can Still See You -- End-to-End Multimodal Object Detection](https://ieeexplore.ieee.org/document/9468959) by Osama Mazhar, Jens Kober and Robert Babuska.

First we performed modifications in the single modal [DETR module](../detr/README.md). To support multimodal inputs and allow different fusion methods, we performed additional modifications in this package.

The modifications are:
- In [transforms.py](algorithm/datasets/transforms.py), the following classes are added: *seededRandomCrop* and *RandomShadows*.
- In [mm_detr.py](algorithm/models/mm_detr.py), the *sc_avg_detr* and *avg_baseline* classes are added in order to allow multimodal inputs.

DETR was originally licensed under the Apache 2.0 [license](https://github.com/facebookresearch/detr/blob/master/LICENSE).

Also, the MobileNetV2 backbone implementation in [backbone_mobilenetv2.py](algorithm/models/backbone_mobilenetv2.py) is based on the [implementation by Zhiqiang Wang](https://github.com/zhiqwang/demonet/blob/dd4cec83abf5bd937ebf3ebc767972431223d33e/demonet/models/backbone.py).
Here, the modifications are:
- The extra blocks are removed from the *BackboneBase* class
- The *InvertedResidual* and *VonvBNActivation* classes are removed
- The *Joiner* class is added

The MobileNetV2 implementation was or originally licensed under the Apache 2.0 [license](https://github.com/zhiqwang/demonet/blob/dd4cec83abf5bd937ebf3ebc767972431223d33e/LICENSE)

All modifications are licensed under the Apache 2.0 [license](../../../../../LICENSE) by OpenDR European Project.

Also, code from [here](https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv) is used in the [drawing utility function](algorithm/util/draw.py).