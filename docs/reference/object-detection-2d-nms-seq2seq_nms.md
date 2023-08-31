## Seq2Seq-NMS module

The *seq2seq-nms* module contains the *Seq2SeqNMSLearner* class, which inherits from the abstract class *Learner* and the abstract class *NMSCustom*.

### Class Seq2SeqNMSLearner
Bases: `engine.learners.Learner` and `perception.object_detection_2d.nms.utils.NMSCustom`

It can be used to perform single-class non-maximum suppression (NMS) on images (inference) as well as training new seq2seq-nms models. The implementation is based on [[1]](#seq2seq_nms-1). The method is set-up for performing NMS on the person-detection task, using the implemention of the [SSD](/docs/reference/object-detection-2d-ssd.md) detector. The Seq2Seq-NMS method can also be employed for performing single-class NMS, in any class other than human/pedestrian class. In that case the method needs to be trained from scratch. Finally, a pretrained-model can be employed for evaluation or inference on the same class that it was trained with, using RoIs from a different detector than the one used in the training. In that case, we advise to fine-tune the Seq2Seq-nms pretrained model using RoIs from the detector, deployed in the inference/evaluation of the method, in order to achieve the highest possible performance.

The [Seq2SeqNMSLearner](/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/seq2seq_nms_learner.py) class has the following
public methods:

#### `Seq2SeqNMSLearner` constructor
```python
Seq2SeqNMSLearner(self, lr, epochs, device, temp_path, checkpoint_after_iter, checkpoint_load_iter, log_after, variant,
                  iou_filtering, dropout, app_feats, fmod_map_type, fmod_map_bin, app_input_dim)
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
- **variant**: *{'light', 'medium', 'full'}, default='medium'*\
  Specifies the variant of seq2seq-nms model.
- **iou_filtering**: *float, default=0.8*\
  Specifies the IoU threshold used for filtering RoIs before provided by the seq2seq-nms model.
  If set to values <0 or >1, no filtering is applied.
- **dropout**: *float, default=0.025*\
  Specifies the dropout rate.
- **app_feats**: *{'fmod', 'zeros', 'custom'}, default='fmod'*\
  Specifies the type of the appearance-based features of RoIs used in the model.
- **fmod_map_type**: *{'EDGEMAP', 'FAST', 'AKAZE', 'BRISK', 'ORB'}, default='EDGEMAP'*\
  Specifies the type of maps used by FMoD, in the case where *app_feats*='fmod'.
- **fmod_map_bin**: *bool, default=True*\
  Specifies whether FMoD maps are binary or not, in the case where *app_feats*='fmod'.
- **app_input_dim**: *int, default=None*\
  Specifies the dimension of appearance-based RoI features.
  In the case where *app_feats*='fmod', the corresponding dimension is automatically computed.


#### `Seq2SeqNMSLearner.fit`
```python
Seq2SeqNMSLearner.fit(self, dataset, logging_path, logging_flush_secs, silent, verbose, nms_gt_iou, max_dt_boxes, datasets_folder, use_ssd, ssd_model, lr_step)
```

This method is used to train the algorithm on a `Dataset_NMS` dataset.
Returns a dictionary containing stats regarding the training process.

Parameters:

- **dataset**: *{'PETS', 'COCO'}*\
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
  Specifies the maximum number of RoIs provided to seq2Seq-nms model as input.
- **datasets_folder**: *str, default='./datasets'*\
  Specifies the path to the folder where the datasets are stored.
- **use_ssd**: *bool, default=False*\
  If set to True, RoIs from SSD are fed to the seq2Seq-nms model.
  Otherwise, RoIs from the default detector of the specified dataset are used as input.
- **ssd_model**: *{'ssd_512_vgg16_atrous_pets', 'ssd_default_person'} , default=None*\
  The name of SSD's pretrained model. Used only if `use_ssd` is set True.
- **lr_step**: *bool, default=True*\
  If True, decays the learning rate at pre-specified epochs by 0.1.
  
#### `Seq2SeqNMSLearner.eval`
```python
Seq2SeqNMSLearner.eval(self, dataset, split, verbose, max_dt_boxes, threshold, datasets_folder, use_ssd, ssd_model)
```

Performs evaluation on a set of dataset.

Parameters:

- **dataset**: *{'PETS', 'COCO'}*\
  Specifies the name of the dataset among those available from training.
- **split**: *{'train', 'val', 'test'} default='test'*\
  Specifies the set of the corresponding dataset where the evaluation will be performed.
- **verbose**: *bool, default=True*\
  If True, enables maximum verbosity.
- **max_dt_boxes**: *int, default=500*\
  Specifies the maximum number of RoIs provided to seq2Seq-nms model as input.
- **threshold**: *float, default=0.0*\
  Specifies the confidence threshold, used for RoI selection after seq2seq-nms rescoring.
- **datasets_folder**: *str, default='./datasets'*\
  Specifies the path to the folder where the datasets are stored.
- **use_ssd**: *bool, default=False*\
  If set to True, RoIs from SSD are fed to the seq2Seq-nms model.
  Otherwise, RoIs from the default detector of the specified dataset are used as input.
- **ssd_model**: *{'ssd_512_vgg16_atrous_pets', 'ssd_default_person'} , default=None*\
  The name of SSD's pretrained model. Used only if `use_ssd` is set True.

#### `Seq2SeqNMSLearner.infer`
```python
Seq2SeqNMSLearner.infer(self, boxes, scores, boxes_sorted, max_dt_boxes, img_res, threshold)
```

Performs non-maximum suppression, using seq2seq-nms.
In the case where FMoD is selected for appearance-based RoI feature computation, FMoD maps are not computed.

Parameters:

- **boxes**: *torch.tensor, default=None*\
  Image coordinates of candidate detection RoIs, expressed as the coordinates of their upper-left and top-down corners (x_min, y_min, x_max, y_max).
  For N candidate detection RoIs, the size of the *torch.tensor* is Nx4.
- **scores**: *torch.tensor, default=None*\
  Specifies the scores of the candidate detection RoIs, assigned previously by a detector.
  For N candidate detection RoIs, the size of the *torch.tensor* is Nx1.
- **boxes_sorted**: *bool, default=False*\
  Specifies whether *boxes* and *scores* are sorted based on *scores* in descending order.
- **max_dt_boxes**: *int, default=400*\
  Specifies the maximum number of detection RoIs that are fed as input to seq2seq-nms model.
- **img_res**: *[int, int], default=None*\
  Specifies the image resolution expressed as [width, height].
- **threshold**: *float, default=0.1*\
  Specifies the score threshold that will determine which RoIs will be kept after seq2seq-nms rescoring. 
  
#### `Seq2SeqNMSLearner.run_nms`
```python
Seq2SeqNMSLearner.run_nms(self, boxes, scores, img, threshold, boxes_sorted, top_k, map)
```

Performs non-maximum suppression, using seq2seq-nms.
It incorporates the full pipeline needed for inference, including the FMoD's edge/interest-point map computation step.

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
  Specifies the maximum number of detection RoIs that are fed as input to seq2seq-nms model.
- **img**: *object*\
  Object of type engine.data.Image.
- **threshold**: *float, default=0.1*\
  Specifies the score threshold that will determine which RoIs will be kept after seq2seq-nms rescoring.
- **map**: *numpy.ndarray, default=None*\
  Feature maps extracted by the detector. This method doesn't utilize this kind of input. 
  
#### `Seq2SeqNMSLearner.save`
```python
Seq2SeqNMSLearner.save(self, path, verbose, optimizer, scheduler, current_epoch, max_dt_boxes)
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
  Specifies the maximum number of detection RoIs that are fed as input to seq2seq-nms model.
  
 

#### `Seq2SeqNMSLearner.load`
```python
Seq2SeqNMSLearner.load(self, path, verbose)
```

Loads a model which was previously saved in OpenDR format at the specified path.

Parameters:

- **path**: *str*\
  Specifies the folder where the model will be loaded from.
- **verbose**: *bool default=False*\
  If True, enables maximum verbosity.
  
  
#### `Seq2SeqNMSLearner.download`
```python
Seq2SeqNMSLearner.download(self, path, model_name, verbose, url)
```

Downloads pretrained models of seq2seq-nms.

Parameters:

Downloads data needed for the various functions of the learner, e.g., pretrained models as well as test data.

Parameters:

- **path**: *str, default=None*\
  Specifies the folder where data will be downloaded.
  If *None*, the *self.temp_path* directory is used instead.
- **model_name**: *{'seq2seq_pets_jpd_pets_fmod', 'seq2seq_pets_ssd_wider_person_fmod', 'seq2seq_pets_ssd_pets_fmod', 'seq2seq_coco_frcn_coco_fmod', 'seq2seq_coco_ssd_wider_person_fmod'}, default='seq2seq_pets_jpd_pets_fmod'*\
  Downloads the specified pretrained seq2seq-nms model.
- **verbose**: *bool default=True*\
  If True, enables maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.
  
#### Examples

* **Training example.**
  To train seq2seq-nms properly, the PETS and COCO datasets are supported as Dataset_NMS types. 

  ```python
  from opendr.perception.object_detection_2d.nms import Seq2SeqNMSLearner
  import os
  OPENDR_HOME = os.environ['OPENDR_HOME']
  
  temp_path = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/seq2seq_nms/tmp'
  datasets_folder = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/datasets'
  
  seq2SeqNMSLearner = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering=0.8, 
                                        app_feats='fmod', checkpoint_after_iter=1,
                                        temp_path=temp_path, epochs=8)
  seq2SeqNMSLearner.fit(dataset='PETS', use_ssd=False, datasets_folder=datasets_folder,
                        logging_path=os.path.join(temp_path, 'logs'), silent=False,
                        verbose=True, nms_gt_iou=0.50, max_dt_boxes=500)
  ```

* **Inference and result drawing example on a test .jpg image using OpenCV.**

  ```python
  from opendr.perception.object_detection_2d.nms import Seq2SeqNMSLearner
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import SingleShotDetectorLearner
  from opendr.perception.object_detection_2d import draw_bounding_boxes
  import os
  OPENDR_HOME = os.environ['OPENDR_HOME']
  temp_path = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/seq2seq_nms/tmp'

  seq2SeqNMSLearner = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering=0.8,
                                        app_feats='fmod', device='cpu',
                                        temp_path=temp_path)
  seq2SeqNMSLearner.download(model_name='seq2seq_pets_ssd_pets_fmod', path=temp_path)
  seq2SeqNMSLearner.load(os.path.join(temp_path, seq2seq_pets_ssd_pets_fmod), verbose=True)
  ssd = SingleShotDetectorLearner(device='cuda')
  ssd.download(".", mode="pretrained")
  ssd.load("./ssd_512_vgg16_atrous_pets", verbose=True)
  img = Image.open(OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/img_temp/frame_0000.jpg')
  if not isinstance(img, Image):
      img = Image(img)
  boxes = ssd.infer(img, threshold=0.3, custom_nms=seq2SeqNMSLearner)
  draw_bounding_boxes(img.opencv(), boxes, class_names=ssd.classes, show=True)
  ```
  
* **Evaluation of pretrained model on PETS dataset.**

  ```python
  from opendr.perception.object_detection_2d import Seq2SeqNMSLearner
  import os
  OPENDR_HOME = os.environ['OPENDR_HOME']
  
  dataset_folder = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/datasets'
  temp_path = OPENDR_HOME + '/projects/python/perception/object_detection_2d/nms/seq2seq_nms/tmp'
  
  seq2SeqNMSLearner = Seq2SeqNMSLearner(iou_filtering=0.8, app_feats='fmod',
                                        temp_path=temp_path, device='cuda')
  seq2SeqNMSLearner.download(model_name='seq2seq_pets_jpd_pets_fmod', path=temp_path)
  seq2SeqNMSLearner.load(os.path.join(temp_path, 'seq2seq_pets_jpd_pets_fmod'), verbose=True)
  seq2SeqNMSLearner.eval(dataset='PETS', split='val', max_dt_boxes=800,
                       datasets_folder=dataset_folder, use_ssd=False, threshold=0.0)
  ```
  
#### Performance Evaluation

TABLE-1: Average Precision (AP) achieved by pretrained models on the person detection task on the validation sets. The maximum number or RoIs, employed for the performance evaluation was set to 800.
|  **Pretrained Model**  | **Dataset** | **Detector** | **Detector's training dataset** | **Type of Appearance-based Features** | **Pre-processing IoU Threshold** | **AP<sub>0.5</sub> on validation set** | **AP<sub>0.5</sub> on testing set** |
|:----------------------:|:-----------:|:------------:|:-------------------------------:|:-------------------------------------:|:--------------------------------:|:----------------------------:|:---------------------------:|
|   seq2seq_pets_jpd_pets_fmod   |     PETS    |      JPD     |            PETS           |                  FMoD                 |                0.8               |             80.2%            |          84.3%         |
|  seq2seq_pets_ssd_wider_person_fmod |     PETS    |      SSD     |        WiderPerson        |                  FMoD                 |                0.8               |             77.4%            |          79.1%         |
|  seq2seq_pets_ssd_pets_fmod    |     PETS    |      SSD     |            PETS           |                  FMoD                 |                0.8               |             87.8%            |          91.2%         |
|  seq2seq_coco_frcn_coco_fmod   |     COCO    |     FRCN     |            COCO           |                  FMoD                 |                 -                |             68.1%\*         |        67.5%\*\*      |
| seq2seq_coco_ssd_wider_person_fmod  |     COCO    |      SSD     |        WiderPerson        |                  FMoD                 |                 -                |             41.8%\*         |        42.4%\*\*        |

\* The minival set was used as validation set.<br>
\*\* The minitest set was used as test set.



#### References
<a name="seq2seq_nms-1" href="https://ieeexplore.ieee.org/abstract/document/10107719">[1]</a>  C. Symeonidis, I. Mademlis, I. Pitas and N. Nikolaidis, "Neural Attention-Driven Non-Maximum Suppression for Person Detection" in IEEE Transactions on Image Processing, vol. 32, pp. 2454-2467, 2023, doi: 10.1109/TIP.2023.3268561.
