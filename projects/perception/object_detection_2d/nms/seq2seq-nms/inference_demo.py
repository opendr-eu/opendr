from opendr.perception.object_detection_2d.nms.seq2seq_nms.Seq2SeqNMSLearner import Seq2SeqNMSLearner
from opendr.engine.data import Image
from opendr.perception.object_detection_2d import SingleShotDetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from multiprocessing import Pool
import os
OPENDR_HOME = os.environ['OPENDR_HOME']

seq2SeqNMSLearner = Seq2SeqNMSLearner(fmod_map_type='EDGEMAP', iou_filtering = 0.8, experiment_name='pets_exp6', use_fmod=True,
                                      fmod_init_path=OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/datasets/PETS/FMoD/pets_edgemap_b.pkl')
seq2SeqNMSLearner.load(OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/temp/pets_exp6/checkpoints/checkpoint_epoch_7', verbose=True)
ssd = SingleShotDetectorLearner(device='cpu')
ssd.download(".", mode="pretrained")
ssd.load("./ssd_default_person", verbose=True)
img = Image.open(OPENDR_HOME + '/projects/perception/object_detection_2d/nms/seq2seq-nms/img_temp/frame_0000.jpg')
if not isinstance(img, Image):
    img = Image(img)
boxes = ssd.infer(img, threshold=0, custom_nms=seq2SeqNMSLearner)
seq2SeqNMSLearner.fMoD.extract_maps(img=img, augm=False)
boxes = seq2SeqNMSLearner.infer(classes=ssd.classes, dets=boxes, boxes_sorted=False, max_dt_boxes=1200, img_res=img.opencv().shape[::-1][1:])
draw_bounding_boxes(img.opencv(), boxes, class_names=ssd.classes, show=True)
