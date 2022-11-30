## YOLOv5DetectorLearner module

The *yolov5* module contains the *YOLOv5DetectorLearner* class, which inherits from the abstract class *Learner*.

### Class YOLOv5DetectorLearner
Bases: `engine.learners.Learner`

The *YOLOv5DetectorLearner* class is a wrapper of the YOLO detector[[1]](#yolo-1)
[Ultralytics implementation](https://github.com/ultralytics/yolov5) based on its availability in the [Pytorch Hub](https://pytorch.org/hub/ultralytics_yolov5/).
It can be used to perform object detection on images (inference only).

The [YOLOv5DetectorLearner](/src/opendr/perception/object_detection_2d/yolov5/yolov5_learner.py) class has the following
public methods:

#### `YOLOv5DetectorLearner` constructor
```python
YOLOv5DetectorLearner(self, model_name, path, device)
```

Constructor parameters:

- **model_name**: *str*\
  Specifies the name of the model to be used. Available models: 
   - 'yolov5n' (46.0% mAP,  1.9M parameters)
   - 'yolov5s' (56.0% mAP,  7.2M parameters)
   - 'yolov5m' (63.9% mAP,  21.2M parameters)
   - 'yolov5l' (67.2% mAP,  46.5M parameters)
   - 'yolov5x' (68.9% mAP,  86.7M parameters)
   - 'yolov5n6'  (50.7% mAP, 3.2M parameters)
   - 'yolov5s6' (63.0% mAP,  16.8M parameters)
   - 'yolov5m6' (69.0% mAP,  35.7 parameters)
   - 'yolov5l6' (71.6% mAP, 76.8M parameters)
   - 'custom' (for custom models, the ```path``` parameter must be set to point to the location of the weights file.)
Note that mAP (0.5) is reported on the [COCO val2017 dataset](https://github.com/ultralytics/yolov5/releases).
- **path**: *str, default=None*\
  For custom-trained models, specifies the path to the weights to be loaded.
- **device**: *{'cuda', 'cpu'}, default='cuda'*
  Specifies the device used for inference.
- **temp_path**: *str, default='.'*\
  Specifies the path to where the weights will be downloaded when using pretrained models.
- **force_reload**: *bool, default=False*\
  Sets the `force_reload` parameter of the pytorch hub `load` method.
  This fixes issues with caching when set to `True`.
  

#### `YOLOv5DetectorLearner.infer`
The `infer` method:
```python
YOLOv5DetectorLearner.infer(self, img)
```

Performs inference on a single image.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image or OpenCV.
- **size**: *int, default=640*\
  Size of image for inference.
  The image is resized to this in both sides before being fed to the model.
  
#### Examples

* Inference and result drawing example on a test .jpg image using OpenCV:
  ```python
  import torch
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import YOLOv5DetectorLearner
  from opendr.perception.object_detection_2d import draw_bounding_boxes

  yolo = YOLOv5DetectorLearner(model_name='yolov5s', device='cpu')

  torch.hub.download_url_to_file('https://ultralytics.com/images/zidane.jpg', 'zidane.jpg')  # download image
  im1 = Image.open('zidane.jpg')  # OpenDR image

  results = yolo.infer(im1)
  draw_bounding_boxes(im1.opencv(), results, yolo.classes, show=True, line_thickness=3)
  ```

#### References
<a name="yolo-1" href="https://ultralytics.com/yolov5">[1]</a> YOLOv5: The friendliest AI architecture you'll ever use.
