from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    @abstractmethod
    def assign(self, bboxes, num_level_bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass
