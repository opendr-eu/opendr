import torch
import logging
from torch import nn
from opendr.perception.multimodal_human_centric.intent_recognition_learner.algorithm.backbones.FusionNets.MULT import MULT
__all__ = ['ModelManager']


class MIA(nn.Module):

    def __init__(self, args):

        super(MIA, self).__init__()

        self.model = MULT(args)

    def forward(self, text_feats, video_feats, audio_feats):

        video_feats, audio_feats = video_feats.float(), audio_feats.float()
        mm_model = self.model(text_feats, video_feats, audio_feats)

        return mm_model


class ModelManager:

    def __init__(self, args):

        self.logger = logging.getLogger(args.logger_name)
        self.device = args.device
        self.model = self._set_model(args)

    def _set_model(self, args):

        model = MIA(args)
        print(self.device)
        model.to(self.device)
        return model
