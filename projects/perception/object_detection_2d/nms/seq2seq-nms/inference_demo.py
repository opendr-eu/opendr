from opendr.perception.object_detection_2d.nms.seq2seq_nms.seq2seq_nms_learner import Seq2SeqNMSLearner
from opendr.engine.data import Image
from opendr.perception.object_detection_2d import SingleShotDetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
import datetime
import os
OPENDR_HOME = os.environ['OPENDR_HOME']

seq2SeqNMSLearner = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering = 0.8, experiment_name='auth_exp51', app_feats = 'fmod',
                                      fmod_init_path=OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/'
                                                                   'datasets/PETS/FMoD/pets_edgemap_b.pkl', device='cpu')
seq2SeqNMSLearner.load(OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/temp/auth_exp51/'
                                     'checkpoints/checkpoint_epoch_7', verbose=True)
ssd = SingleShotDetectorLearner(device='cuda')
ssd.download(".", mode="pretrained")
ssd.load("./ssd_default_person", verbose=True)
img = Image.open(OPENDR_HOME + '/projects/perception/object_detection_2d/nms/img_temp/frame_0000.jpg')
if not isinstance(img, Image):
    img = Image(img)
boxes = ssd.infer(img, threshold=0.6, custom_nms=seq2SeqNMSLearner)
draw_bounding_boxes(img.opencv(), boxes, class_names=ssd.classes, show=True)
