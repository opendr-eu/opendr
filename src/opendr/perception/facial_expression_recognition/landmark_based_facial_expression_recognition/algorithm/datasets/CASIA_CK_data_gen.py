# Copyright 2020-2021 OpenDR European Project
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

import os
import numpy as np
import torch
import pickle
import argparse
import pandas
from pathlib import Path


CK_CLASSES = pandas.read_csv(Path(__file__).parent / 'CK+_labels.csv', verbose=True, index_col=0).to_dict()["name"]
CASIA_CLASSES = pandas.read_csv(Path(__file__).parent / 'CASIA_labels.csv', verbose=True, index_col=0).to_dict()["name"]


def data_normalization(data):
    data = torch.from_numpy(data)
    N, V, C, T, M = data.size()
    data = data.permute(0, 2, 3, 1, 4).contiguous().view(N, C, T, V, M)
    # remove the first 17 points
    data = data[:, :, :, 17:, :]
    N, C, T, V, M = data.size()
    # normalization
    for n in range(N):
        for t in range(T):
            for v in range(V):
                data[n, :, t, v, :] = data[n, :, t, v, :] - data[n, :, t, 16, :]
    return data.numpy()


def data_gen(landmark_path, num_frames, num_landmarks, num_dim, num_faces, classes):
    subject_data = []
    lbl_ind = -1
    subject_lbls = []
    subject_lbl_names = []
    for c in classes:
        class_path = os.path.join(landmark_path, c)
        lbl_ind += 1
        lbl_name = c
        if os.path.exists(class_path):
            root, _, files = os.walk(class_path)
            T = len(files)
            if T > 0:  # if the sample video is a nonzero sample
                sample_numpy = np.zeros((num_landmarks, num_dim, num_frames, num_faces))
                for j in range(num_frames-1):
                    if os.path.isfile(class_path + str(T - j - 1) + '.npy'):
                        sample_numpy[:, :, -1 - j, 0] = np.load(class_path + str(T - j - 1) + '.npy')
                for j in range(T):
                    if os.path.isfile(class_path + str(j) + '.npy'):
                        sample_numpy[:, :, 0, 0] = np.load(class_path + str(j) + '.npy')
                        break
                subject_data.append(sample_numpy)
                subject_lbls.append(lbl_ind)
                subject_lbl_names.append(lbl_name)
    subject_data = np.concatenate(subject_data, axis=0)
    return subject_data, subject_lbls, subject_lbl_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facial landmark extractor')
    parser.add_argument("-i", "--landmark_folder", required=True, default='./data/CASIA_landmarks/',
                        description="path to extracted landmarks")
    parser.add_argument("-o", "--output_folder", required=True, default='./data/CASIA_10fold/',
                        description="path to 10fold data")
    parser.add_argument("-data", "--dataset_name", required=True, default='CASIA',
                        description="the name of dataset which can be CASIA or CK+")
    arg = vars(parser.parse_args())
    if arg.dataset_name == 'CASIA':
        classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Disgust']
    elif arg.dataset_name == 'CK+':
        classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Disgust', 'Neutral']
    num_frames = 5
    num_landmarks = 68
    num_dim = 2
    num_faces = 1
    data = []
    labels = []
    label_names = []
    num_subjects = len(os.listdir(arg.landmark_folder))
    for s in range(num_subjects):
        landmark_path = arg.landmark_folder + '/{}/'.format(s)
        subject_data, subject_lbls, subject_lbl_names = data_gen(landmark_path, num_frames, num_landmarks,
                                                                 num_dim, num_faces, classes)
        data.append(subject_data)
        labels.append(subject_lbls)
        label_names.append(subject_lbl_names)

    # 10 fold data generation
    fold_size = round(num_subjects/10)
    folds_list = folds_lbl_list = folds_lbl_names_list = [[], [], [], [], [], [], [], [], [], []]
    for i in range(10):
        fold_data = []
        fold_lbl = []
        fold_lbl_name = []
        down = i * fold_size
        up = (i + 1) * fold_size
        if i == 9:
            up = num_subjects
        for j in range(down, up):
            fold_data = fold_data.append(data[j])
            fold_lbl = fold_lbl+labels[j]
            fold_lbl_name = fold_lbl_name+label_names[j]
        fold_data = np.concatenate(fold_data, axis=0)
        fold_data = data_normalization(fold_data)
        folds_list[i] = fold_data
        folds_lbl_list[i] = fold_lbl
        folds_lbl_names_list[i] = fold_lbl_name
    # generate train and val data
    for f in range(9, -1, -1):
        lbl = []
        lbl_name = []
        if f > 0:
            data = np.concatenate((folds_list[0:f]), axis=0)
            for k in range(0, f):
                lbl = lbl + folds_lbl_list[k]
                lbl_name = lbl_name + folds_lbl_names_list[k]
            if f + 1 < 10:
                tmp = np.concatenate((folds_list[f+1:]), axis=0)
                data = np.concatenate((data, tmp), axis=0)
                for k in range(f + 1, 10):
                    lbl = lbl + folds_lbl_list[k]
                    lbl_name = lbl_name + folds_lbl_names_list[k]
        elif f == 0:
            data = np.concatenate((folds_list[f + 1:]), axis=0)
            for k in range(f + 1, 10):
                lbl = lbl + folds_lbl_list[k]
                lbl_name = lbl_name + folds_lbl_names_list[k]
        Train = data
        Val = folds_list[f]
        Train_lbl = (lbl_name, lbl)
        Val_lbl = (folds_lbl_names_list[f], folds_lbl_list[f])
        np.save(arg.output_folder + '_train_' + f + '.npy', Train)
        with open(arg.output_folder + '_train_labels_' + f + '.pkl', 'wb') as l:
            pickle.dump(Train_lbl, l)
        np.save(arg.output_folder + '_val_' + f + '.npy', Val)
        with open(arg.output_folder + '_val_labels_' + f + '.pkl', 'wb') as l:
            pickle.dump(Val_lbl, l)
