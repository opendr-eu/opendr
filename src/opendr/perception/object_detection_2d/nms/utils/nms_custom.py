from abc import ABC, abstractmethod


class NMSCustom(ABC):
    def __init__(self, device='cpu'):
        self.device = device

    @abstractmethod
    def run_nms(self, boxes=None, scores=None, threshold=0.2, img=None, device='cpu'):
        pass
