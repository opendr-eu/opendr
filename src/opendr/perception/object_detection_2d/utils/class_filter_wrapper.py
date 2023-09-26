from opendr.engine.target import BoundingBox, BoundingBoxList


class FilteredLearnerWrapper:
    def __init__(self, learner, allowed_classes=None):
        self.learner = learner
        self.allowed_classes = allowed_classes if allowed_classes is not None else []

    def infer(self, img, threshold=0.1, keep_size=True):

        boxes = self.learner.infer(img, threshold=threshold, keep_size=keep_size)
        if not self.allowed_classes:
            return boxes
        else:
            obj_det_classes = self.learner.classes

            # Check if all allowed classes exist in the list of object detector's classes
            invalid_classes = [cls for cls in self.allowed_classes if cls not in obj_det_classes]
            if invalid_classes:
                raise ValueError(
                    f"The following classes are not detected by this detector: {', '.join(invalid_classes)}")

            filtered_boxes = BoundingBoxList(
                [box for box in boxes if obj_det_classes[int(box.name)] in self.allowed_classes])

        return filtered_boxes

    def __getattr__(self, attr):
        return getattr(self.learner, attr)
