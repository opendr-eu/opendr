from opendr.perception.object_detection_2d.nms.cluster_nms.cluster_nms import ClusterNMS
from opendr.engine.data import Image
from opendr.perception.object_detection_2d import SingleShotDetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
import datetime
import os
OPENDR_HOME = os.environ['OPENDR_HOME']

ssd = SingleShotDetectorLearner(device='cuda')
ssd.download(".", mode="pretrained")
ssd.load("./ssd_default_person", verbose=True)
img = Image.open(OPENDR_HOME + '/projects/perception/object_detection_2d/nms/seq2seq-nms/img_temp/frame_0000.jpg')
if not isinstance(img, Image):
    img = Image(img)
start_time = datetime.datetime.now()
cluster_nms = ClusterNMS(device='cuda', nms_type='default', cross_class=True)
boxes = ssd.infer(img, threshold=0.3, custom_nms=cluster_nms)
draw_bounding_boxes(img.opencv(), boxes, class_names=ssd.classes, show=True)