# Copyright 2020-2022 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parts of this code is modified from https://github.com/zengqunzhao/EfficientFace/blob/master/models/EfficientFace.py:
"""

# MIT License
#
# Copyright (c) 2021 Zengqun (Zeke) Zhao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm.efficientface_modulator import Modulator
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm.efficientface_utils import (
     LocalFeatureExtractor,
     InvertedResidual
     )
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm.transformer_timm import (
     AttentionBlock,
     Attention
     )


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    padding = kernel_size//2
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True))


class EfficientFaceTemporal(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, num_classes=8, im_per_sample=15):
        super(EfficientFaceTemporal, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.local = LocalFeatureExtractor(29, 116, 1)
        self.modulator = Modulator(116)

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                                   nn.BatchNorm2d(output_channels),
                                   nn.ReLU(inplace=True),)
        self.conv1d_0 = conv1d_block(output_channels, 64)
        self.conv1d_1 = conv1d_block(64, 64)

        self.conv1d_2 = conv1d_block(64, 128)
        self.conv1d_3 = conv1d_block(128, 128)

        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )

        self.im_per_sample = im_per_sample

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.modulator(self.stage2(x)) + self.local(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        return x

    def forward_stage1(self, x):
        assert x.shape[0] % self.im_per_sample == 0, "Batch size is not a multiple of sequence length."
        n_samples = x.shape[0] // self.im_per_sample
        x = x.view(n_samples, self.im_per_sample, x.shape[1])
        x = x.permute(0, 2, 1)
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x


def init_feature_extractor(model, path):
    if path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    model.load_state_dict(pre_trained_dict, strict=False)


def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0),
                         nn.BatchNorm1d(out_channels),
                         nn.ReLU(inplace=True), nn.MaxPool1d(2, 1))


class AudioCNNPool(nn.Module):

    def __init__(self, num_classes=8):
        super(AudioCNNPool, self).__init__()
        input_channels = 10
        self.conv1d_0 = conv1d_block_audio(input_channels, 64)
        self.conv1d_1 = conv1d_block_audio(64, 128)
        self.conv1d_2 = conv1d_block_audio(128, 256)
        self.conv1d_3 = conv1d_block_audio(256, 128)

        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x

    def forward_stage1(self, x):
        x = self.conv1d_0(x)
        x = self.conv1d_1(x)
        return x

    def forward_stage2(self, x):
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        return x

    def forward_classifier(self, x):
        x = x.mean([-1])  # pooling accross temporal dimension
        x = self.classifier(x)
        return x


class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=8, fusion='ia', seq_length=15, pretr_ef=None, num_heads=1):
        super(MultiModalCNN, self).__init__()

        self.audio_model = AudioCNNPool(num_classes=num_classes)
        self.visual_model = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes, seq_length)
        self.fusion = fusion

        init_feature_extractor(self.visual_model, pretr_ef)

        e_dim = 128
        input_dim_video = 64
        input_dim_audio = 128
        if fusion == 'lt':
            input_dim_video = 128
            self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads)
            self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads)
        elif fusion == 'it':
            self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio,
                                     out_dim=input_dim_audio, num_heads=num_heads)
            self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video,
                                     out_dim=input_dim_video, num_heads=num_heads)
        elif fusion == 'ia':
            self.av = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio,
                                out_dim=input_dim_audio, num_heads=num_heads)
            self.va = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video,
                                out_dim=input_dim_video, num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(e_dim*2, num_classes),
        )

    def forward(self, x_audio, x_visual):

        if self.fusion == 'lt':
            return self.forward_late_transformer(x_audio, x_visual)
        elif self.fusion == 'ia':
            return self.forward_int_attention(x_audio, x_visual)
        elif self.fusion == 'it':
            return self.forward_int_transformer(x_audio, x_visual)

    def forward_int_transformer(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.va(proj_x_a, proj_x_v)

        h_av = h_av.permute(0, 2, 1)
        h_va = h_va.permute(0, 2, 1)

        x_audio = h_av+x_audio
        x_visual = h_va + x_visual

        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
        video_pooled = x_visual.mean([-1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)

        x = self.classifier(x)
        return x

    def forward_int_attention(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)

        proj_x_a = x_audio.permute(0, 2, 1)
        proj_x_v = x_visual.permute(0, 2, 1)

        _, h_av = self.av(proj_x_v, proj_x_a)
        _, h_va = self.va(proj_x_a, proj_x_v)

        if h_av.size(1) > 1:  # if more than 1 head, take mean
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)

        h_av = h_av.sum([-2])

        if h_va.size(1) > 1:  # if more than 1 head, take mean
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)

        h_va = h_va.sum([-2])

        x_audio = h_va*x_audio
        x_visual = h_av*x_visual

        x_audio = self.audio_model.forward_stage2(x_audio)
        x_visual = self.visual_model.forward_stage2(x_visual)

        audio_pooled = x_audio.mean([-1])  # mean accross temporal dimension
        video_pooled = x_visual.mean([-1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)

        x = self.classifier(x)
        return x

    def forward_late_transformer(self, x_audio, x_visual):
        x_audio = self.audio_model.forward_stage1(x_audio)
        proj_x_a = self.audio_model.forward_stage2(x_audio)

        x_visual = self.visual_model.forward_features(x_visual)
        x_visual = self.visual_model.forward_stage1(x_visual)
        proj_x_v = self.visual_model.forward_stage2(x_visual)

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        h_av = self.av(proj_x_v, proj_x_a)
        h_va = self.va(proj_x_a, proj_x_v)

        audio_pooled = h_av.mean([1])  # mean accross temporal dimension
        video_pooled = h_va.mean([1])

        x = torch.cat((audio_pooled, video_pooled), dim=-1)

        x = self.classifier(x)
        return x
