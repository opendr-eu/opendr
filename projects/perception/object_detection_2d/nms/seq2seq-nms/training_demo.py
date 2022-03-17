from opendr.perception.object_detection_2d.nms.seq2seq_nms.seq2seq_nms_learner import Seq2SeqNMSLearner
import os
OPENDR_HOME = os.environ['OPENDR_HOME']
temp_path = OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/seq2seq_nms/temp/coco_own3'
seq2SeqNMSLearner = Seq2SeqNMSLearner(iou_filtering=None, app_feats='fmod',
                                      temp_path=temp_path, device='cuda')
seq2SeqNMSLearner.load(temp_path + '/checkpoints/checkpoint_epoch_0', verbose=True)
seq2SeqNMSLearner.eval(dataset='COCO', split='val', max_dt_boxes=800,
                       datasets_folder=OPENDR_HOME + '/src/opendr/perception/object_detection_2d/nms/datasets',
                       use_ssd=False)
