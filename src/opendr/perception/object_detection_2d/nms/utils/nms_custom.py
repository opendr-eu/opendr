from abc import ABC, abstractmethod


class NMSCustom(ABC):
    @abstractmethod
    def run_nms(self, boxes=None, scores=None, threshold=0.2, img=None, device='cpu'):
        pass
