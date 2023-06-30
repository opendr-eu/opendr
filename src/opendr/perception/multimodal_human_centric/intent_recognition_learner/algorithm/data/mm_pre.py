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
import csv
import torch
import numpy as np
from .text_pre import TextDataset
from .video_pre import VideoDataset
from .audio_pre import AudioDataset
from .__init__ import benchmarks
from opendr.engine.datasets import DatasetIterator


class MIntRecDataset(DatasetIterator):

    def __init__(
            self,
            data_path,
            video_data_path,
            audio_data_path,
            text_backbone,
            cache_path='cache',
            padding_mode='zero',
            padding_loc='end',
            split='train'):
        assert split in ['train', 'dev', 'test']
        super(MIntRecDataset, self).__init__()
        self.cache_path = cache_path
        self.data_path = os.path.join(data_path, 'MIntRec')
        self.benchmarks = benchmarks['MIntRec']
        self.label_list = self.benchmarks["intent_labels"]
        self.data_index, label_ids = self._get_indexes_annotations(os.path.join(self.data_path, '{}.tsv'.format(split)))

        attrs = self._get_attrs()
        text_feats = TextDataset(text_backbone, attrs, split).feats
        video_feats = VideoDataset(video_data_path, padding_mode, padding_loc, attrs, split).feats
        audio_feats = AudioDataset(audio_data_path, padding_mode, padding_loc, attrs, split).feats

        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.video_feats = torch.tensor(video_feats)
        self.audio_feats = torch.tensor(np.array(audio_feats))
        self.size = len(self.text_feats)

    def _get_attrs(self):
        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value
        return attrs

    def _get_indexes_annotations(self, read_file_path):
        label_map = {}
        for i, label in enumerate(self.label_list):
            label_map[label] = i
        with open(read_file_path, 'r') as f:
            data = csv.reader(f, delimiter="\t")
            indexes = []
            label_ids = []
            for i, line in enumerate(data):
                if i == 0:
                    continue

                index = '_'.join([line[0], line[1], line[2]])
                indexes.append(index)

                label_id = label_map[line[4]]
                label_ids.append(label_id)
        return indexes, label_ids

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = {
            'label_ids': self.label_ids[index],
            'text_feats': self.text_feats[index],
            'video_feats': self.video_feats[index],
            'audio_feats': self.audio_feats[index]
        }
        return sample
