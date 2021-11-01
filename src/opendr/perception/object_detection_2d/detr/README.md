# OpenDR Object Detection 2D - DETR

This folder contains an implementation of DETR in an OpenDR Learner class for 2D object detection. The input of the model is an image and the output consists of bounding boxes with class names and confidences.

## Sources

Large parts of the code are taken from [facebook/detr](https://github.com/facebookresearch/detr) with modifications to make them compatible with OpenDR specifications. The original DETR paper can be found here: [End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko.

The modifications are:
- Instead of the [main.py](https://github.com/facebookresearch/detr/blob/master/main.py) file for creating, training and evaluating models, we created an OpenDR learner class with these functionalities. Some code inside the learner class was copied from [main.py](https://github.com/facebookresearch/detr/blob/master/main.py), especially in the fit method of the learner. Changes that were made are:
  - Many variables are class attributes instead of local variables.
  - Functionality for verbose and silent mode are added.
  - The functionality for loading from a checkpoint using the load_from_iter attribute is added.
- The rest of the algorithm code is mainly copied from [facebook/detr](https://github.com/facebookresearch/detr), but with the following modifications:
  - The names of the coco and coco_panoptic dataset subfolders for the images and annotations are not fixed any more.
  - The number of classes in [detr.py](https://github.com/facebookresearch/detr/blob/master/models/detr.py) is not fixed any more.
  - Verbose and silent mode are added to the [engine.py](https://github.com/facebookresearch/detr/blob/master/engine.py) and metric_logger in [misc.py](https://github.com/facebookresearch/detr/blob/master/util/misc.py)

  DETR was originally licensed under the Apache 2.0 [license](https://github.com/facebookresearch/detr/blob/master/LICENSE).

  The modifications are also licensed under the Apache 2.0 license by OpenDR European Project.

Also, code from [here](https://stackoverflow.com/questions/60674501/how-to-make-black-background-in-cv2-puttext-with-python-opencv) is used in the [drawing utility function](algorithm/util/draw.py).