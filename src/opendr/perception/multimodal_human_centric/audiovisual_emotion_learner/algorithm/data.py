# Copyright 2020-2023 OpenDR European Project
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

import torch
import numpy as np
import os
from opendr.engine.datasets import DatasetIterator
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm import spatial_transforms as transforms
from opendr.perception.multimodal_human_centric.audiovisual_emotion_learner.algorithm.data_utils import (
     preprocess_video,
     preprocess_audio
     )
import librosa
from PIL import Image
import random
from tqdm import tqdm


class RavdessDataset(DatasetIterator):
    def __init__(self, annotation, video_transform, sr=22050, n_mfcc=10):

        self.annotation = annotation
        self.video_transform = video_transform
        self.sr = sr
        self.n_mfcc = n_mfcc

    def __len__(self,):
        return len(self.annotation)

    def __getitem__(self, i):

        target = self.annotation[i]['label']
        video = np.load(self.annotation[i]['video_path'])
        video = [Image.fromarray(video[i, :, :, :])
                 for i in range(np.shape(video)[0])]
        self.video_transform.randomize_parameters()
        video = [self.video_transform(img) for img in video]
        video = torch.stack(video, 0).permute(1, 0, 2, 3)

        audio = np.load(self.annotation[i]['audio_path']).astype(np.float32)
        audio = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)

        return audio, video, target


class DataWrapper:
    def __init__(self, opendr_dataset):
        self.dataset = opendr_dataset

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y, z = self.dataset.__getitem__(i)
        return x.data, y.data, z.data


def parse_annotations(path, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
    train_dataset = []
    val_dataset = []
    test_dataset = []
    for line in annots:
        videofilename, audiofilename, label, trainvaltest = line.rstrip().split(';')
        videofilename = os.path.join(path, videofilename)
        audiofilename = os.path.join(path, audiofilename)

        assert os.path.exists(videofilename), 'File {} not found.'.format(videofilename)
        assert os.path.exists(audiofilename), 'File {} not found.'.format(audiofilename)

        sample = {'video_path': videofilename,
                  'audio_path': audiofilename,
                  'label': int(label)-1}

        if trainvaltest == 'training':
            train_dataset.append(sample)
        elif trainvaltest == 'testing':
            test_dataset.append(sample)
        elif trainvaltest == 'validation':
            val_dataset.append(sample)

    return train_dataset, val_dataset, test_dataset


def get_random_split_ravdess():
    ids = list(np.arange(1, 25))
    s1 = ids[::2]
    s2 = ids[1::2]
    random.shuffle(s1)
    random.shuffle(s2)
    n_train = 8
    n_val = 2
    train_ids = s1[:n_train] + s2[:n_train]
    val_ids = s1[n_train:n_train+n_val]+s2[n_train:n_train+n_val]
    test_ids = s1[n_train+n_val:] + s2[n_train+n_val:]
    return train_ids, val_ids, test_ids


def preprocess_ravdess(src='RAVDESS/', sr=22050, n_mfcc=10, target_time=3.6,
                       input_fps=30, save_frames=15, target_im_size=224, device='cpu'):
    train_ids, val_ids, test_ids = get_random_split_ravdess()
    annotations_file = os.path.join(src, 'annotations.txt')
    for actor in os.listdir(src):
        if int(actor[-2:]) in train_ids:
            subset = 'training'
        elif int(actor[-2:]) in val_ids:
            subset = 'validation'
        elif int(actor[-2:]) in test_ids:
            subset = 'testing'

        for file in tqdm(os.listdir(os.path.join(src, actor))):
            if file.endswith('.mp4'):
                video = preprocess_video(os.path.join(src, actor, file), target_time, input_fps,
                                         save_frames, target_im_size, device)
                np.save(os.path.join(src, actor, file.replace('.mp4', '.npy')), video)
                label = str(int(file.split('-')[2]))
                audio_path = '03' + file[2:].replace('.mp4', '.wav')
                audio = preprocess_audio(os.path.join(src, actor, audio_path), sr, target_time)
                np.save(os.path.join(src, actor, audio_path.replace('.wav', '.npy')), audio)
                with open(annotations_file, 'a') as f:
                    f.write(os.path.join(src, actor, file.replace('.mp4', '.npy')) +
                            ';' + os.path.join(src, actor, audio_path.replace('.wav', '.npy')) +
                            ';' + label + ';' + subset + '\n')


def get_audiovisual_emotion_dataset(path='RAVDESS/', sr=22050, n_mfcc=10, preprocess=False,
                                    target_time=3.6, input_fps=30, save_frames=15, target_im_size=224, device='cpu'):
    if preprocess:
        preprocess_ravdess(path, sr, n_mfcc, target_time, input_fps, save_frames, target_im_size, device)
    annot_path = os.path.join(path, 'annotations.txt')

    train_annots, val_annots, test_annots = parse_annotations(path, annot_path)
    video_scale = 255

    video_train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(video_scale)])

    video_val_transform = transforms.Compose([
                transforms.ToTensor(video_scale)])

    train_set = RavdessDataset(train_annots, video_train_transform, sr=sr, n_mfcc=n_mfcc)
    val_set = RavdessDataset(val_annots, video_val_transform, sr=sr, n_mfcc=n_mfcc)
    test_set = RavdessDataset(test_annots, video_val_transform, sr=sr, n_mfcc=n_mfcc)
    return train_set, val_set, test_set
