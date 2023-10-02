## YOLOv8SegLearner module

The *yolov8_seg* module contains the *YOLOv8SegLearner* class, which inherits from the abstract class *Learner*.

### Class YOLOv8SegLearner
Bases: `engine.learners.Learner`

The *YOLOv8SegLearner* class is a wrapper of the YOLOv8 detector[[1]](#yolo-1)
[Ultralytics implementation](https://github.com/ultralytics/ultralytics), for its semantic segmentation variant.
It can be used to perform semantic segmentation on images (inference only). The tool can also return bounding boxes.
The detected classes can be seen 
[here](https://github.com/ultralytics/ultralytics/blob/9aaa5d5ed0e5a0c1f053069dd73f12b845c4f282/ultralytics/cfg/datasets/coco.yaml#L17).

Note that to be compatible with the OpenDR Heatmap format, we modify the `person` class to be index `80` instead of `0`, with
`0` being the background (pixels with no detections).

The [YOLOv8SegLearner](/src/opendr/perception/semantic_segmentation/yolov8_seg/yolov8_seg_learner.py) class has the following
public methods:

#### `YOLOv8SegLearner` constructor
```python
YOLOv8SegLearner(self, model_name, model_path, device, temp_path)
```

Constructor parameters:

- **model_name**: *str*\
  Specifies the name of the model to be used. Available models: 
   - 'yolov8n-seg' (30.5% mAP(mask),  3.4M parameters)
   - 'yolov8s-seg' (36.8% mAP(mask),  11.8M parameters)
   - 'yolov8m-seg' (40.8% mAP(mask),  27.3M parameters)
   - 'yolov8l-seg' (42.6% mAP(mask),  46.0M parameters)
   - 'yolov8x-seg' (43.4% mAP(mask),  71.8M parameters)
   - 'custom'  (for custom models, the `model_path` parameter must be set to point to the location of the `.pt` file.)
Note that mAP (0.5-0.95) is reported on [the official website](https://docs.ultralytics.com/models/yolov8/#supported-modes)
for the segmentation task on the COCO2017 val dataset.
- **model_path**: *str, default=None*\
  For custom-trained models, specifies the path to the `.pt` file.
- **device**: *{'cuda', 'cpu'}, default='cuda'*
  Specifies the device used for inference.
- **temp_path**: *str, default='.'*\
  Specifies the path to where the pretrained model will be downloaded.  

#### `YOLOv8SegLearner.infer`
The `infer` method:
```python
YOLOv8SegLearner.infer(self, img, conf_thres, iou_thres, image_size, half_prec, agnostic_nms, classes, no_mismatch)
```

Performs inference on a single image. Various arguments of the original implementation are exposed.

Parameters:

- **img**: *object*\
  Object of type engine.data.Image or OpenCV image.
- **conf_thres**: *float, default=0.25*\
  Object confidence threshold for detection.
- TODO
- **image_size**: *int or tuple, default=None*\
  Image size as scalar or (h, w) list, i.e. (640, 480).
  
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
<a name="yolo-1" href="https://ultralytics.com/yolov8">[1]</a> YOLOv8: The State-of-the-Art YOLO Model
