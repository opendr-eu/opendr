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
  Specifies the name of the model to be used. See `YOLOv5DetectorLearner.available_models` for a list of available architectures.
- **path**: *str, default=None*\
  For custom-trained models, specifies the path to the weights to be loaded.
- **device**: *{'cuda', 'cpu'}, default='cuda'*
  Specifies the device used for inference.
- **temp_path**: *str, default=None*\
  Specifies the path to where the weights will be downloaded when using pretrained models.
  

#### `YOLOv5DetectorLearner.infer`
The `infer` method:
```python
YOLOv5DetectorLearner.infer(self, img)
```

performs inference on a single image.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image or OpenCV.
  
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
  draw_bounding_boxes(im1.opencv(), results, yolo.classes, show=False)
  ```

#### References
<a name="yolo-1" href="https://ultralytics.com/yolov5">[1]</a> YOLOv5: The friendliest AI architecture you'll ever use.
