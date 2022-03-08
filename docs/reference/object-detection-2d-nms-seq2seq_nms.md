## Seq2Seq-NMS module

The *seq2seq-nms* module contains the *Seq2SeqNMSLearner* class, which inherits from the abstract class *Learner*.

### Class Seq2SeqNMSLearner
Bases: `engine.learners.Learner`


It can be used to perform single-class non-maximum suppression on images (inference) as well as training new seq2seq-nms models.

The [Seq2SeqNMSLearner](/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/seq2seq_nms_learner.py) class has the following
public methods:

#### `Seq2SeqNMSLearner` constructor
```python
Seq2SeqNMSLearner(self, lr, epochs, device, temp_path, checkpoint_after_iter, checkpoint_load_iter, log_after, variant, experiment_name,
                 iou_filtering, dropout, pretrained_demo_model, app_feats, fmod_map_type, fmod_map_bin, app_input_dim, fmod_init_path)
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
  Specifies the epoch interval between checkpoints during training. If set to 0 no checkpoint will be saved.
- **checkpoint_load_iter**: *int, default=0*\
  Specifies the epoch to load a saved checkpoint from. If set to 0 no checkpoint will be loaded.
- **log_after**: *int, default=500*\
  Specifies interval (in iterations/batches) between information logging on *stdout*.
- **variant**: *{'light', 'medium', 'full'}, default='medium'*\
  Specifies the variant of seq2seq-nms model.
- **experiment_name**: *str, default='default'*\
  Specifies the name of the experiment.
- **iou_filtering**: *float, default=0.8*\
  Specifies the IoU threshold used for filtering RoIs before provided by the seq2seq-nms model.If set to values <0 or >1, no filtering is applied.
- **dropout**: *float, default=0.05*\
  Specifies the dropout rate.
- **pretrained_demo_model**: *{'PETS_JPD_medium', 'COCO_FRCN_medium' , defualt=None*\
- Specifies the name of the pretrained model
- **app_feats**: *{'fmod', 'zeros', 'custom'}, default='fmod'*\
  Specifies the type of the appearance-based features of RoIs used in the model.
- **fmod_map_type**: *{'EDGEMAP', 'FAST', 'AKAZE', 'BRISK', 'ORB'}, default='EDGEMAP'*\
  Specifies the type of maps used by FMoD, in the case where *app_feats*='fmod'.
- **fmod_map_bin**: *bool, default=True*\
  Specifies whether FMoD maps are binary or not, in the case where *app_feats*='fmod'.
- **app_input_dim**: *int, default=None*\
  Specifies the dimension of appearance-based RoI features. In the case where *app_feats*='fmod', the corresponding dimension is automatically computed.
- **fmod_init_path**: *str, default=None *\
  Specifies the path to the the file used for normalizing appearance-based features, in the case where *app_feats*='fmod'.
