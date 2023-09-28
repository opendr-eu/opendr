from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.perception.object_detection_2d.centernet.centernet_learner import CenterNetDetectorLearner
from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner
from opendr.perception.object_detection_2d.gem.gem_learner import GemLearner
from opendr.perception.object_detection_2d.retinaface.retinaface_learner import RetinaFaceLearner
from opendr.perception.object_detection_2d.ssd.ssd_learner import SingleShotDetectorLearner
from opendr.perception.object_detection_2d.yolov3.yolov3_learner import YOLOv3DetectorLearner
from opendr.perception.object_detection_2d.yolov5.yolov5_learner import YOLOv5DetectorLearner
from opendr.perception.object_detection_2d.nanodet.nanodet_learner import NanodetLearner


class FilteredLearnerWrapper:
    def __init__(self, learner, allowed_classes=None):
        self.learner = learner
        self.allowed_classes = allowed_classes if allowed_classes is not None else []

        if isinstance(self.learner,
                      (CenterNetDetectorLearner, YOLOv3DetectorLearner, YOLOv5DetectorLearner, NanodetLearner,
                       RetinaFaceLearner, SingleShotDetectorLearner)):
            self.classes = self.learner.classes
        if isinstance(self.learner, (DetrLearner, GemLearner)):
            coco_classes = [
                "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "trafficlight", "firehydrant", "streetsign", "stopsign", "parkingmeter", "bench", "bird",
                "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat",
                "backpack", "umbrella", "shoe", "eyeglasses", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard",
                "surfboard", "tennisracket", "bottle", "plate", "wineglass", "cup", "fork", "knife",
                "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog",
                "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "mirror", "diningtable",
                "window", "desk", "toilet", "door", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
                "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush", "hairbrush"
            ]
            self.classes = coco_classes

        # Check if all allowed classes exist in the list of object detector's classes
        invalid_classes = [cls for cls in self.allowed_classes if cls not in self.classes]
        if invalid_classes:
            raise ValueError(
                f"The following classes are not detected by this detector: {', '.join(invalid_classes)}")

    def infer(self, img=None, threshold=None, keep_size=None, m1_image=None, m2_image=None, input=None,
              conf_threshold=None, iou_threshold=None, nms_max_num=None, nms_threshold=None, scales=None,
              mask_thresh=None, size=None, custom_nms=None, nms_thresh=None, nms_topk=None, post_nms=None,
              extract_maps=None):

        if isinstance(self.learner, GemLearner):
            if m1_image is None or m2_image is None:
                raise ValueError(
                    f"Two image inputs are required. Please provide two valid images.")
            boxes = self.learner.infer(m1_image, m2_image)

        elif isinstance(self.learner, NanodetLearner):
            if input is None:
                raise ValueError(
                    f"An image input is required. Please provide a valid image.")
            if conf_threshold is None:
                conf_threshold = 0.35
            if iou_threshold is None:
                iou_threshold = 0.6
            if nms_max_num is None:
                nms_max_num = 100
            boxes = self.learner.infer(img, conf_threshold, iou_threshold, nms_max_num)

        else:
            if img is None:
                raise ValueError(
                    f"An image input is required. Please provide a valid image.")

            if isinstance(self.learner, CenterNetDetectorLearner):
                if threshold is None:
                    threshold = 0.2
                if keep_size is None:
                    keep_size = True
                boxes = self.learner.infer(img, threshold=threshold, keep_size=keep_size)

            elif isinstance(self.learner, YOLOv3DetectorLearner):
                if threshold is None:
                    threshold = 0.1
                if keep_size is None:
                    keep_size = True
                boxes = self.learner.infer(img, threshold=threshold, keep_size=keep_size)

            elif isinstance(self.learner, YOLOv5DetectorLearner):
                if size is None:
                    size = 640
                boxes = self.learner.infer(img, size)

            elif isinstance(self.learner, DetrLearner):
                boxes = self.learner.infer(img)

            elif isinstance(self.learner, RetinaFaceLearner):
                if threshold is None:
                    threshold = 0.8
                if nms_threshold is None:
                    nms_threshold = 0.4
                if scales is None:
                    scales = [1024, 1980]
                if mask_thresh is None:
                    mask_thresh = 0.8
                boxes = self.learner.infer(img, threshold, nms_threshold, scales, mask_thresh)

            elif isinstance(self.learner, SingleShotDetectorLearner):
                if threshold is None:
                    threshold = 0.2
                if keep_size is None:
                    keep_size = False
                if nms_thresh is None:
                    nms_thresh = 0.45
                if nms_topk is None:
                    nms_topk = 400
                if post_nms is None:
                    post_nms = 100
                if extract_maps is None:
                    extract_maps = False
                boxes = self.learner.infer(img, threshold, keep_size, custom_nms,
                                           nms_thresh, nms_topk, post_nms, extract_maps)

        if not self.allowed_classes:
            return boxes
        else:
            filtered_boxes = BoundingBoxList(
                [box for box in boxes if self.classes[int(box.name)] in self.allowed_classes])

            return filtered_boxes

    def __getattr__(self, attr):
        return getattr(self.learner, attr)
