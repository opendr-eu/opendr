
# OpenDR Multimodal Object Detection 2D - GEM

This folder contains an implementation of GEM Multimodal Object Detector in an OpenDR Learner class for 2D object detection. The input of the model is a list of two images, one from each modality e.g., RGB and Infrared, and the output consists of bounding boxes with class names and confidences.

## Sources

This module currently depends on OpenDR's DETR module which is a wrapper of [facebook/detr](https://github.com/facebookresearch/detr) with modifications to make them compatible with OpenDR specifications. The GEM paper can be found here: [GEM: Glare or Gloom, I Can Still See You -- End-to-End Multimodal Object Detection](https://arxiv.org/abs/2102.12319) by Osama Mazhar, Jens Kober and Robert Babuska.

First we performed modifications in the single modal DETR which are detailed in `opendr.perception.object_detection_2d.detr` package. To support multimodal inputs and allow different fusion methods, we performed additional modifications in this package.

The modifications are:
- (to be updated)

DETR was originally licensed under the Apache 2.0 [license](https://github.com/facebookresearch/detr/blob/master/LICENSE).

The modifications are also licensed under the Apache 2.0 license by OpenDR European Project.
