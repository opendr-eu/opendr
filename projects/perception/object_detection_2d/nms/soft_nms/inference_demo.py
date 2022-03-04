from opendr.perception.object_detection_2d.nms.soft_nms.soft_nms import SoftNMS
from opendr.engine.data import Image
from opendr.perception.object_detection_2d import SingleShotDetectorLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
import os
OPENDR_HOME = os.environ['OPENDR_HOME']

ssd = SingleShotDetectorLearner(device='cuda')
ssd.download(".", mode="pretrained")
ssd.load("./ssd_default_person", verbose=True)
img = Image.open(OPENDR_HOME + '/projects/perception/object_detection_2d/nms/seq2seq-nms/img_temp/frame_0000.jpg')
if not isinstance(img, Image):
    img = Image(img)
cluster_nms = SoftNMS(device='cpu', nms_type='gaussian')
boxes = ssd.infer(img, threshold=0.3, custom_nms=cluster_nms)
draw_bounding_boxes(img.opencv(), boxes, class_names=ssd.classes, show=True)