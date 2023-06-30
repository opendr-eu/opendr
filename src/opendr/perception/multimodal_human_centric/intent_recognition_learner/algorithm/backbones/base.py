'''
This code is based on https://github.com/thuiar/MIntRec under MIT license:

MIT License

Copyright (c) 2022 Hanlei Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
'''


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
