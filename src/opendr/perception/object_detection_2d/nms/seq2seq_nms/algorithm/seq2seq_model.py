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

import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Seq2SeqNet(nn.Module):
    def __init__(self, dropout=0.01, use_app_feats=True, app_input_dim=315, geom_input_dim=14, lq_dim=256, sq_dim=128,
                 num_JPUs=4, device='cuda'):
        super().__init__()
        self.use_app_feats = use_app_feats
        self.dropout_q = nn.Dropout(dropout * 0.25)
        self.num_JPUs = num_JPUs
        self.joint_processing_units = []
        self.device = device
        for i in range(self.num_JPUs):
            self.joint_processing_units.append(Joint_processing_unit(lq_dim=lq_dim, sq_dim=sq_dim, dropout=dropout))
            if "cuda" in self.device:
                self.joint_processing_units[i] = self.joint_processing_units[i].to(self.device)
        self.joint_processing_units = nn.ModuleList(self.joint_processing_units)
        if self.use_app_feats:
            q_app_dims = [180, 180]
            self.q_app_layers = nn.Sequential(
                nn.Linear(app_input_dim, q_app_dims[0]),
                nn.GELU(),
                nn.Dropout(dropout * 0.25),
                nn.LayerNorm(q_app_dims[0], eps=1e-6),
                nn.Linear(q_app_dims[0], q_app_dims[1]),
                nn.GELU(),
                nn.Dropout(dropout * 0.25),
                # nn.LayerNorm(q_fmod_dims[1], eps=1e-6)
            )

        q_geom_dims = [180, 180]
        self.q_geom_layers = nn.Sequential(
            nn.Linear(geom_input_dim, q_geom_dims[0]),
            nn.GELU(),
            nn.LayerNorm(q_geom_dims[0], eps=1e-6),
            nn.Linear(q_geom_dims[0], q_geom_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            # nn.LayerNorm(q_geom_dims[1], eps=1e-6)
        )

        k_geom_dims = [180, 180]
        self.k_geom_layers = nn.Sequential(
            nn.Linear(geom_input_dim, k_geom_dims[0]),
            nn.GELU(),
            nn.LayerNorm(k_geom_dims[0], eps=1e-6),
            nn.Linear(k_geom_dims[0], k_geom_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            # nn.LayerNorm(k_geom_dims[1], eps=1e-6)
        )

        q_final_in_dim = q_geom_dims[-1]
        k_final_in_dim = k_geom_dims[-1]
        if self.use_app_feats:
            q_final_in_dim = q_geom_dims[-1] + q_app_dims[-1]
            k_final_in_dim = k_geom_dims[-1] + q_app_dims[-1]

        self.q_full_layers = nn.Sequential(
            nn.LayerNorm(q_final_in_dim, eps=1e-6),
            nn.Linear(q_final_in_dim, lq_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            # nn.LayerNorm(lq_dim, eps=1e-6)
        )
        self.k_full_layers = nn.Sequential(
            nn.LayerNorm(k_final_in_dim, eps=1e-6),
            nn.Linear(k_final_in_dim, sq_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            # nn.LayerNorm(sq_dim, eps=1e-6)
        )
        self.q_final_layers = nn.Sequential(
            nn.LayerNorm(lq_dim, eps=1e-6),
            nn.Linear(lq_dim, sq_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.LayerNorm(sq_dim, eps=1e-6),
            nn.Linear(sq_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, q_geom_feats=None, k_geom_feats=None, msk=None, app_feats=None):
        q_feats = self.q_geom_layers(q_geom_feats)
        k_feats = self.k_geom_layers(k_geom_feats)

        if self.use_app_feats and app_feats is not None:
            app_feats = self.q_app_layers(app_feats)
            q_feats = torch.cat((q_feats, app_feats), dim=2)
            k_feats = torch.cat((k_feats, app_feats.transpose(0, 1).repeat(k_feats.shape[1], 1, 1)), dim=2)

        elif app_feats is None:
            raise UserWarning("Appearance-based representations not provided.")
        q_feats = self.q_full_layers(q_feats)
        k_feats = self.k_full_layers(k_feats)
        for i in range(self.num_JPUs):
            q_feats, k_feats = self.joint_processing_units[i](q_feats, k_feats, msk)
        scores = self.q_final_layers(q_feats)
        return scores.squeeze(1)


class Joint_processing_unit(nn.Module):
    def __init__(self, heads=2, lq_dim=256, sq_dim=128, dropout=0.1):
        super().__init__()
        self.q_block1 = nn.Sequential(
            nn.LayerNorm(lq_dim, eps=1e-6),
            nn.Linear(lq_dim, sq_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.norm_layer_q = nn.LayerNorm(sq_dim, eps=1e-6)
        self.norm_layer_k = nn.LayerNorm(sq_dim, eps=1e-6)
        self.self_attention_module = Self_attention_module(heads=heads, l_dim=lq_dim, s_dim=sq_dim, dropout=dropout)
        self.scale_layer = Scale_layer(s_dim=sq_dim)

        self.q_block2 = nn.Sequential(
            nn.LayerNorm(sq_dim, eps=1e-6),
            nn.Linear(sq_dim, lq_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, q_feats, k_feats, msk):
        q_atten = self.q_block1(q_feats)
        kv_atten_in = self.norm_layer_k(k_feats)
        q_atten_in = self.norm_layer_q(q_atten)
        q_atten = q_atten + self.self_attention_module(q=q_atten_in, k=kv_atten_in, v=kv_atten_in, mask=msk)
        k_feats = k_feats + self.scale_layer(q_atten).transpose(0, 1).repeat(q_atten.shape[0], 1, 1)
        q_feats = q_feats + self.q_block2(q_atten)
        return q_feats, k_feats


class Self_attention_module(nn.Module):
    def __init__(self, heads, l_dim, s_dim, dropout=0.1):
        super().__init__()
        self.l_dim = l_dim
        self.s_dim = s_dim
        self.qkv_split_dim = s_dim // heads
        self.h = heads
        self.q_linear = nn.Linear(self.s_dim, self.s_dim)
        self.v_linear = nn.Linear(self.s_dim, self.s_dim)
        self.k_linear = nn.Linear(self.s_dim, self.s_dim)

        self.dropout = nn.Dropout(dropout)
        self.q_out = nn.Sequential(
            nn.Linear(self.s_dim, self.s_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v, mask=None):
        samples_dim = q.size(0)
        k = self.k_linear(k).view(samples_dim, -1, self.h, self.qkv_split_dim).transpose(1, 2)
        q = self.q_linear(q).view(samples_dim, -1, self.h, self.qkv_split_dim).transpose(1, 2)
        v = self.v_linear(v).view(samples_dim, -1, self.h, self.qkv_split_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.qkv_split_dim)

        mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, scores.shape[1], 1, 1)
        scores = torch.mul(scores, mask)
        scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        q = torch.matmul(scores, v)
        q = q.transpose(1, 2).contiguous().view(samples_dim, -1, self.s_dim)
        q = self.q_out(q)
        return q


class Scale_layer(nn.Module):
    def __init__(self, s_dim=1):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.empty(s_dim), requires_grad=True)
        nn.init.uniform_(self.scale_weights, a=0.01, b=2.0)

    def forward(self, feats):
        return feats * self.scale_weights
