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


import torch
import torch.nn.functional as F
from ..SubNets.FeatureNets import LanguageModel
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from torch import nn

__all__ = ['MULT']

_TEXT_DIMS = {'bert-base-uncased': 768,
              'albert-base-v2': 768,
              'prajjwal1/bert-small': 512,
              'prajjwal1/bert-mini': 256,
              'prajjwal1/bert-tiny': 128
              }


class MULT(nn.Module):

    def __init__(self, args):

        super(MULT, self).__init__()
        assert args.text_backbone in [
            'bert-base-uncased',
            'albert-base-v2',
            'prajjwal1/bert-tiny',
            'prajjwal1/bert-mini',
            'prajjwal1/bert-small']
        self.text_subnet = LanguageModel(args)
        video_feat_dim = args.video_feat_dim
        text_feat_dim = _TEXT_DIMS[args.text_backbone]
        audio_feat_dim = args.audio_feat_dim

        dst_feature_dims = args.dst_feature_dims
        self.orig_d_l, self.orig_d_a, self.orig_d_v = text_feat_dim, audio_feat_dim, video_feat_dim
        self.d_l = self.d_a = self.d_v = dst_feature_dims

        self.num_heads = args.nheads
        self.layers = args.n_levels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v

        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        assert args.mode in ['audio', 'video', 'language', 'joint', 'mult']
        self.mode = args.mode

        self.combined_dim = combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = args.num_labels

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        if self.mode in ['mult', 'joint']:
            self.trans_l_with_a = self._get_network(self_type='la')
            self.trans_l_with_v = self._get_network(self_type='lv')

            self.trans_a_with_l = self._get_network(self_type='al')
            self.trans_a_with_v = self._get_network(self_type='av')

            self.trans_v_with_l = self._get_network(self_type='vl')
            self.trans_v_with_a = self._get_network(self_type='va')

            self.trans_l_mem = self._get_network(self_type='l_mem', layers=3)
            self.trans_a_mem = self._get_network(self_type='a_mem', layers=3)
            self.trans_v_mem = self._get_network(self_type='v_mem', layers=3)

            self.proj1 = nn.Linear(combined_dim, combined_dim)
            self.proj2 = nn.Linear(combined_dim, combined_dim)
            self.out_layer = nn.Linear(combined_dim, output_dim)
        if self.mode in ['audio', 'joint']:
            self.transformer_audio_head = self._get_network(self_type='a')
            self.audio_class = nn.Linear(combined_dim, output_dim)
            self.audio_fin_proj = nn.Linear(self.d_a, combined_dim)
        if self.mode in ['video', 'joint']:
            self.transformer_video_head = self._get_network(self_type='v')
            self.video_class = nn.Linear(combined_dim, output_dim)
            self.video_fin_proj = nn.Linear(self.d_v, combined_dim)
        if self.mode in ['language', 'joint']:
            self.transformer_language_head = self._get_network(self_type='l')
            self.language_class = nn.Linear(combined_dim, output_dim)
            self.language_fin_proj = nn.Linear(self.d_l, combined_dim)

    def _get_network(self, self_type='l', layers=-1):

        if self_type in ['l', 'vl', 'al']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x_l, x_v, x_a):
        if self.mode == 'mult':
            return self.forward_multimodal(x_l, x_a, x_v)
        elif self.mode == 'audio':
            return self.forward_audio(x_a)
        elif self.mode == 'video':
            return self.forward_video(x_v)
        elif self.mode == 'language':
            return self.forward_language(x_l)
        elif self.mode == 'joint':
            return self.forward_joint(x_l, x_a, x_v)

    def forward_audio(self, x_a, return_attention=False):
        x_a = x_a.transpose(1, 2)

        proj_x_a = self.proj_a(x_a)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        audio, attention = self.transformer_audio_head(proj_x_a)
        audio = audio[0]  # .mean(0)
        audio = self.audio_fin_proj(audio)
        audio = nn.ReLU()(audio)
        out = self.audio_class(audio)
        if return_attention:
            return out, proj_x_a, attention
        return out, proj_x_a

    def forward_video(self, x_v, return_attention=False):
        x_v = x_v.transpose(1, 2)

        proj_x_v = self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        video, attention = self.transformer_video_head(proj_x_v)
        video = video[0]  # .mean(0)
        video = self.video_fin_proj(video)
        video = nn.ReLU()(video)
        out = self.video_class(video)
        if return_attention:
            return out, proj_x_v, attention
        return out, proj_x_v

    def forward_language(self, x_l, return_attention=False):
        x_l = self.text_subnet(x_l)
        x_l = x_l.transpose(1, 2)

        proj_x_l = self.proj_l(x_l)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        language, attention = self.transformer_language_head(proj_x_l)
        language = language[0]  # .mean(0)
        language = self.language_fin_proj(language)
        language = nn.ReLU()(language)
        out = self.language_class(language)
        if return_attention:
            return out, proj_x_l, attention
        return out, proj_x_l

    def forward_joint(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        audio_out, proj_x_a, attention_audio_dist = self.forward_audio(x_a, return_attention=True)
        video_out, proj_x_v, attention_video_dist = self.forward_video(x_v, return_attention=True)
        language_out, proj_x_l, attention_language_dist = self.forward_language(x_l, return_attention=True)

        h_l_with_as, _ = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs, _ = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls, attention_language = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[0]  # .mean(0)

        h_a_with_ls, _ = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs, _ = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as, attention_audio = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = last_hs = h_as[0]  # .mean(0)

        h_v_with_ls, _ = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as, _ = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs, attention_video = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[0]  # .mean(0)

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        last_hs_proj = F.relu(self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.output_dropout, training=self.training)))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, audio_out, video_out, language_out, attention_audio, \
            attention_video, attention_language, attention_audio_dist, \
            attention_video_dist, attention_language_dist

    def forward_multimodal(self, x_l, x_a, x_v):
        x_l = self.text_subnet(x_l)

        x_l = x_l.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        h_l_with_as, _ = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs, _ = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)

        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls, _ = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[0]  # .mean(0)
        print(last_h_l.shape)
        h_a_with_ls, _ = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs, _ = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as, _ = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = last_hs = h_as[0]  # .mean(0)

        h_v_with_ls, _ = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as, _ = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs, _ = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[0]  # .mean(0)

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        last_hs_proj = F.relu(self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.output_dropout, training=self.training)))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs
