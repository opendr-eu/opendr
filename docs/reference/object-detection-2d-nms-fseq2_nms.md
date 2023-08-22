## FSeq2-NMS module

The *fseq2-nms* module contains the *FSeq2NMSLearner* class, which inherits from the abstract class *Learner* and the abstract class *NMSCustom*.

### Class FSeq2NMSLearner
Bases: `engine.learners.Learner` and `perception.object_detection_2d.nms.utils.NMSCustom`

It can be used to perform single-class non-maximum suppression (NMS) on images (inference) as well as training new fseq2-nms models. The implementation is based on [[1]](#fseq2_nms-1) [[2]](#fseq2_nms-2). The method is set-up for performing NMS on the person-detection task, using the implemention of the [SSD](/docs/reference/object-detection-2d-ssd.md) detector. The FSeq2-NMS method can also be employed for performing single-class NMS, in any class other than human/pedestrian class. In that case the method needs to be trained from scratch. Finally, a pretrained-model can be employed for evaluation or inference on the same class that it was trained with, using RoIs from a different detector than the one used in the training. In that case, we advise to fine-tune the FSeq2-nms pretrained model using RoIs from the detector, deployed in the inference/evaluation of the method, in order to achieve the highest possible performance.

The [FSeq2NMSLearner](/src/opendr/perception/object_detection_2d/nms/fseq2_nms/fseq2_nms_learner.py) class has the following
public methods:

#### `FSeq2NMSLearner` constructor
```python
FSeq2NMSLearner(self, lr, epochs, device, temp_path, checkpoint_after_iter, checkpoint_load_iter, log_after, 
                iou_filtering, dropout, app_input_dim)
```

Constructor parameters:

- **lr**: *float, default=0.0001*\
  Specifies the initial learning rate to be used during training.
- **epochs**: *int, default=8*\
  Specifies the number of epochs to be used during training.
- **device**: *{'cuda', 'cpu'}, default='cuda'*\
  Specifies the device to be used.
- **temp_path**: *str, default='./temp'*\
  Specifies a path to be used for storage of checkpoints during training.
- **checkpoint_after_iter**: *int, default=0*\
  Specifies the epoch interval between checkpoints during training.
  If set to 0 no checkpoint will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies the epoch to load a saved checkpoint from.
  If set to 0 no checkpoint will be loaded.
- **log_after**: *int, default=500*\
  Specifies interval (in iterations/batches) between information logging on *stdout*.
- **iou_filtering**: *float, default=0.8*\
  Specifies the IoU threshold used for filtering RoIs before provided by the fseq2-nms model.
  If set to values <0 or >1, no filtering is applied.
- **dropout**: *float, default=0.025*\
  Specifies the dropout rate.
- **app_input_dim**: *int, default=315*\
  Specifies the dimension of appearance-based RoI features.

#### `FSeq2NMSLearner.fit`
```python
FSeq2NMSLearner.fit(self, dataset, logging_path, logging_flush_secs, silent, verbose, nms_gt_iou, max_dt_boxes,
                    datasets_folder, use_ssd, ssd_model, lr_step)
```

This method is used to train the algorithm on a `Dataset_NMS` dataset.
Returns a dictionary containing stats regarding the training process.

Parameters:

- **dataset**: *{'PETS', 'CROWDHUMAN'}*\
  Specifies the name of the dataset among those available from training.
- **logging_path**: *str, default=None*\
  Path to save log files.
  If set to None, only the console will be used for logging.
- **logging_flush_secs**: *int, default=30*\
  How often, in seconds, to flush the TensorBoard data to disk.
- **silent**: *bool, default=False*\
  If set to True, disables all printing of training progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*\
  If True, enables maximum verbosity.
- **nms_gt_iou**: *float, default=0.5*\
  Specifies the threshold used to determine whether a detection RoI must be suppressed or not based on its IoU with the image's ground-truth RoIs.
- **max_dt_boxes**: *int, default=500*\
  Specifies the maximum number of RoIs provided to fseq2-nms model as input.
- **datasets_folder**: *str, default='./datasets'*\
  Specifies the path to the folder where the datasets are stored.
- **use_ssd**: *bool, default=True*\
  If set to True, RoIs from SSD are fed to the fseq2-nms model.
  Otherwise, RoIs from the default detector of the specified dataset are used as input.
- **ssd_model**: *{'ssd_512_vgg16_atrous_pets', 'ssd_default_person'} , default=None*\
  The name of SSD's pretrained model. Used only if `use_ssd` is set True.
- **lr_step**: *bool, default=True*\
  If True, decays the learning rate at pre-specified epochs by 0.1.

#### `FSeq2NMSLearner.eval`
```python
FSeq2NMSLearner.eval(self, dataset, split, verbose, max_dt_boxes, datasets_folder)
```

Performs evaluation on a set of dataset.

Parameters:

- **dataset**: *{'PETS', 'TEST_MODULE}*\
  Specifies the name of the dataset among those available from training.
- **split**: *{'train', 'val', 'test'} default='test'*\
  Specifies the set of the corresponding dataset where the evaluation will be performed.
- **verbose**: *bool, default=True*\
  If True, enables maximum verbosity.
- **max_dt_boxes**: *int, default=400*\
  Specifies the maximum number of RoIs provided to fseq2-nms model as input.
- **threshold**: *float, default=0.0*\
  Specifies the confidence threshold, used for RoI selection after fseq2-nms rescoring.
- **datasets_folder**: *str, default='./datasets'*\
  Specifies the path to the folder where the datasets are stored.
- **use_ssd**: *bool, default=False*\
  If set to True, RoIs from SSD are fed to the seq2Seq-nms model.
  Otherwise, RoIs from the default detector of the specified dataset are used as input.
- **ssd_model**: *{'ssd_512_vgg16_atrous_pets', 'ssd_default_person'} , default=None*\
  The name of SSD's pretrained model. Used only if `use_ssd` is set True.

#### `FSeq2NMSLearner.infer`
```python
FSeq2NMSLearner.infer(self, map, boxes, scores, boxes_sorted, max_dt_boxes, img_res, threshold)
```

Performs non-maximum suppression, using fseq2-nms.

Parameters:

- **map**: *torch.Tensor, default=None*\
  Feature maps extracted by the detector, used as input in Fseq2-NMS.
- **boxes**: *torch.tensor, default=None*\
  Image coordinates of candidate detection RoIs, expressed as the coordinates of their upper-left and top-down corners (x_min, y_min, x_max, y_max).
  For N candidate detection RoIs, the size of the *torch.tensor* is Nx4.
- **scores**: *torch.tensor, default=None*\
  Specifies the scores of the candidate detection RoIs, assigned previously by a detector.
  For N candidate detection RoIs, the size of the *torch.tensor* is Nx1.
- **boxes_sorted**: *bool, default=False*\
  Specifies whether *boxes* and *scores* are sorted based on *scores* in descending order.
- **max_dt_boxes**: *int, default=400*\
  Specifies the maximum number of detection RoIs that are fed as input to fseq2-nms model.
- **img_res**: *[int, int], default=None*\
  Specifies the image resolution expressed as [width, height].
- **threshold**: *float, default=0.1*\
  Specifies the score threshold that will determine which RoIs will be kept after fseq2-nms rescoring. 
  Specifies the maximum number of detection RoIs that are fed as input to fseq2-nms model. 

#### `FSeq2NMSLearner.run_nms`
```python
FSeq2NMSLearner.run_nms(self, boxes, scores, img, threshold, boxes_sorted, top_k, maps)
```

Performs non-maximum suppression, using fseq2-nms.
Wrapper method of *infer*\ method.

Parameters:

- **boxes**: *numpy.ndarray, default=None*\
  Image coordinates of candidate detection RoIs, expressed as the coordinates of their upper-left and top-down corners (x_min, y_min, x_max, y_max).
  For N candidate detection RoIs, the size of the array is Nx4.
- **scores**: *numpy.ndarray, default=None*\
  Specifies the scores of the candidate detection RoIs, assigned previously by a detector.
  For N candidate detection RoIs, the size of the array is Nx1.
- **boxes_sorted**: *bool, default=False*\
  Specifies whether *boxes* and *scores* are sorted based on *scores* in descending order.
- **top_k**: *int, default=400*\
  Specifies the maximum number of detection RoIs that are fed as input to fseq2-nms model.
- **img**: *object*\
  Object of type engine.data.Image.
- **threshold**: *float, default=0.1*\
  Specifies the score threshold that will determine which RoIs will be kept after fseq2-nms rescoring. 
- **map**: *numpy.ndarray, default=None*\
  Feature maps extracted by the detector, used as input in Fseq2-NMS.

#### `FSeq2NMSLearner.save`
```python
FSeq2NMSLearner.save(self, path, verbose, optimizer, scheduler, current_epoch, max_dt_boxes)
```

Saves a model in OpenDR format at the specified path. 

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be saved.
- **verbose**: *bool default=False*\
  If True, enables maximum verbosity.
- **optimizer**: *torch.optim.Optimizer default=None*\
  Specifies the optimizer used for training.
- **scheduler**: *torch.optim.lr_scheduler default=None*\
  Specifies the learning rate scheduler used for training.
- **current_epoch**: *int, default=None*\
  Specifies the number of epochs the model has been trained.
- **max_dt_boxes**: *int, default=400*\
  Specifies the maximum number of detection RoIs that are fed as input to fseq2-nms model.

#### `FSeq2NMSLearner.load`
```python
FSeq2NMSLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be loaded from.
- **verbose**: *bool default=False*\
  If True, enables maximum verbosity.
  

#### `FSeq2NMSLearner.download`
```python
FSeq2NMSLearner.download(self, path, model_name, verbose, url)
```

Downloads pretrained models of fseq2-nms.

Parameters:

Downloads data needed for the various functions of the learner, e.g., pretrained models as well as test data.

Parameters:

- **path**: *str, default=None*\
  Specifies the folder where data will be downloaded.
  If *None*, the *self.temp_path* directory is used instead.
- **model_name**: *{fseq2_pets_ssd_pets'}, default='fseq2_pets_ssd_pets'*\
  Downloads the specified pretrained fseq2-nms model.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.
  
#### Examples

* **Training example.**
  To train fseq2-nms properly, the PETS dataset is supported as Dataset_NMS types. 

  ```python
  from opendr.perception.object_detection_2d.nms import FSeq2NMSLearner
  import os
  OPENDR_HOME = os.environ['OPENDR_HOME']
  
  temp_path = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/fseq2_nms/tmp'
  datasets_folder = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/datasets'
  ssd_model = 'ssd_512_vgg16_atrous_pets'
  fSeq2NMSLearner = FSeq2NMSLearner(iou_filtering=0.8, checkpoint_after_iter=1, temp_path=temp_path, epochs=8)
  fSeq2NMSLearner.fit(dataset='PETS', datasets_folder=datasets_folder, logging_path=os.path.join(temp_path, 'logs'),
                      silent=False, verbose=True, nms_gt_iou=0.50, max_dt_boxes=500, ssd_model=ssd_model)
  ```

* **Inference and result drawing example on a test .jpg image using OpenCV.**

  ```python
  from opendr.perception.object_detection_2d.nms import FSeq2NMSLearner
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import SingleShotDetectorLearner
  from opendr.perception.object_detection_2d.utils.vis_utils import draw_bounding_boxes
  import os
  
  OPENDR_HOME = os.environ['OPENDR_HOME']
  temp_path = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/tmp'
  ssd_model = 'ssd_512_vgg16_atrous_pets'

  fseq2NMSLearner = FSeq2NMSLearner(iou_filtering = 0.8, device='cpu', temp_path=temp_path)
  fseq2NMSLearner.download(model_name='fseq2_pets_ssd_pets', path=temp_path)
  fseq2NMSLearner.load(os.path.join(temp_path, 'fseq2_pets_ssd_pets'), verbose=True)
  ssd = SingleShotDetectorLearner(device='cuda')
  ssd.download(".", mode="pretrained")
  ssd.load(ssd_model, verbose=True)
  img = Image.open(OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/img_temp/frame_0000.jpg')
  if not isinstance(img, Image):
    img = Image(img)
  boxes, _ = ssd.infer(img=img, threshold=0.23, custom_nms=fseqNMSLearner)
  draw_bounding_boxes(img=img.opencv(), bounding_boxes=boxes, class_names=ssd.classes, show=True)
  ```
  
* **Evaluation of pretrained model on PETS dataset.**

  ```python
  from opendr.perception.object_detection_2d import FSeq2NMSLearner
  import os
  OPENDR_HOME = os.environ['OPENDR_HOME']
  
  datasets_folder = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/datasets'
  temp_path = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/tmp'
  ssd_model = 'ssd_512_vgg16_atrous_pets'
  fSeq2NMSLearner = FSeq2NMSLearner(iou_filtering=0.8, temp_path=temp_path, device='cuda')
  fSeq2NMSLearner.download(model_name='fseq2_pets_ssd_pets', path=temp_path)
  fSeq2NMSLearner.load(os.path.join(temp_path, 'fseq2_pets_ssd_pets'), verbose=True)
  fSeq2NMSLearner.eval(dataset='PETS', split='test', max_dt_boxes=800,
                       datasets_folder=datasets_folder, ssd_model=ssd_model, threshold=0.0)
  ```
  
#### Performance Evaluation

TABLE-1: Parameters of the performed evaluation.  
|||
|:-----------:|:----------------------:|
|  **Dataset**   |   PETS  |
|  **Detector**  |   SSD   |
| **Detector's training dataset** | PETS |
| **Pre-processing IoU Threshold** | 0.8 |
| **Maximum number of outputted RoIs** | 800 |

TABLE-2: Average Precision (AP) achieved by pretrained models on the person detection task on the validation and testing sets.
| **Method**  |  **model_name / nms_threshold**  | **AP<sub>0.5</sub> on validation set** | **AP<sub>0.5</sub><sup>0.95</sup> on validation set** |**AP<sub>0.5</sub> on testing set** | **AP<sub>0.5</sub><sup>0.95</sup> on testing set** |
|:-----------:|:--------------------------------:|:--------------------------------------:|:-----------------------------------------------------:|:----------------------------------:|:--------------------------------------------------:|
|         Fast-NMS           |     nms_thres: 0.70     |             81.9%            |          34.9%         |             87.4%            |          37.0%         |
|    Soft-NMS<sub>L</sub>    |     nms_thres: 0.55     |             85.5%            |          37.1%         |             90.4%            |          39.2%         |
|    Soft-NMS<sub>G</sub>    |     nms_thres: 0.90     |             84.2%            |          37.3%         |             90.0%            |          39.6%         |
|        Cluster-NMS         |     nms_thres: 0.60     |             84.6%            |          36.0%         |             90.2%            |          38.2%         |
|  Cluster-NMS<sub>S</sub>   |     nms_thres: 0.35     |             85.1%            |          37.1%         |             90.3%            |          39.0%         |
|  Cluster-NMS<sub>D</sub>   |     nms_thres: 0.55     |             84.8%            |          35.7%         |             90.5%            |          38.1%         |
| Cluster-NMS<sub>S+D</sub>  |     nms_thres: 0.45     |             86.0%            |          37.2%         |             90.9%            |          39.2%         |
| Cluster-NMS<sub>S+D+W</sub>|     nms_thres: 0.45     |             86.0%            |          37.2%         |             90.9%            |          39.2%         |
|        Seq2Seq-NMS         | name: seq2seq_pets_ssd  |             87.8%            |          38.4%         |             91.2%            |          39.5%         |
|        Fseq2-NMS           |  name: fseq2_pets_ssd   |             87.8%            |          38.6%         |             91.5%            |          39.4%         |



#### References
<a name="fseq2_nms-1" href="[object-detection-2d-nms-fseq2_nms.md](object-detection-2d-nms-fseq2_nms.md)https://ieeexplore.ieee.org/abstract/document/10095074">[1]</a>  C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "Neural Attention-Driven Non-Maximum Suppression for Person Detection" in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10095074.

<a name="fseq2_nms-2" href="[object-detection-2d-nms-seq2seq_nms.md](object-detection-2d-nms-seq2seq_nms.md)https://ieeexplore.ieee.org/abstract/document/10107719">[2]</a>  C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "Neural Attention-Driven Non-Maximum Suppression for Person Detection" in IEEE Transactions on Image Processing, vol. 32, pp. 2454-2467, 2023, doi: 10.1109/TIP.2023.3268561.
