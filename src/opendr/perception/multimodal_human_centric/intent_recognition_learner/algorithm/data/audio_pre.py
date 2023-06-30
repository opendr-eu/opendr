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


import os
import numpy as np
import pickle

__all__ = ['AudioDataset']


class AudioDataset:

    def __init__(self, audio_data_path, padding_mode, padding_loc, base_attrs, split):

        audio_feats_path = os.path.join(base_attrs['data_path'], audio_data_path, 'audio_feats.pkl')

        if not os.path.exists(audio_feats_path):
            raise Exception('Error: The directory of audio features is empty.')

        self.feats = self.__load_feats(audio_feats_path, base_attrs)

        self.feats = self.__padding_feats(padding_mode, padding_loc, base_attrs)

    def __load_feats(self, audio_feats_path, base_attrs):

        with open(audio_feats_path, 'rb') as f:
            audio_feats = pickle.load(f)

        feats = [audio_feats[x] for x in base_attrs['data_index']]
        return feats

    def __padding(self, feat, audio_max_length, padding_mode='zero', padding_loc='end'):
        """
        padding_mode: 'zero' or 'normal'
        padding_loc: 'start' or 'end'
        """
        assert padding_mode in ['zero', 'normal']
        assert padding_loc in ['start', 'end']

        audio_length = feat.shape[0]
        if audio_length >= audio_max_length:
            return feat[audio_max_length, :]

        if padding_mode == 'zero':
            pad = np.zeros([audio_max_length - audio_length, feat.shape[-1]])
        elif padding_mode == 'normal':
            mean, std = feat.mean(), feat.std()
            pad = np.random.normal(mean, std, (audio_max_length - audio_length, feat.shape[1]))

        if padding_loc == 'start':
            feat = np.concatenate((pad, feat), axis=0)
        else:
            feat = np.concatenate((feat, pad), axis=0)

        return feat

    def __padding_feats(self, padding_mode, padding_loc, base_attrs):

        audio_max_length = base_attrs['benchmarks']['max_seq_length_audio']

        padding_feats = []

        for feat in self.feats:
            feat = np.array(feat)
            padding_feat = self.__padding(feat, audio_max_length, padding_mode=padding_mode, padding_loc=padding_loc)
            padding_feats.append(padding_feat)
        return padding_feats
