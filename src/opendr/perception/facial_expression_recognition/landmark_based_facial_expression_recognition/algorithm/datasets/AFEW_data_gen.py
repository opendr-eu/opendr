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
import argparse
import pickle


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def data_normalization(data):
    Data = torch.from_numpy(data)
    N, V, C, T, M = Data.size()
    Data = Data.permute(0, 2, 3, 1, 4).contiguous().view(N, C, T, V, M)
    # remove the first 17 points
    Data = Data[:, :, :, 17:, :]
    N, C, T, V, M = Data.size()
    # normalization
    for n in range(N):
        for t in range(T):
            for v in range(V):
                Data[n, :, t, v, :] = Data[n, :, t, v, :] - Data[n, :, t, 16, :]
    Data = Data.numpy()
    return Data


def afew_data_gen(landmark_path, c, num_frames, num_landmarks, num_dim, num_faces):
    class_samples = os.listdir(landmark_path)
    num_samples = len(class_samples)
    class_numpy = np.zeros((num_samples, num_landmarks, num_dim, num_frames, num_faces))
    s = 0
    for _, dirs, _ in os.walk(landmark_path):
        root, _, files = os.walk(dirs)
        T = len(files)
        if T > 0:  # if the sample video is a nonzero sample
            sample_numpy = np.zeros((num_landmarks, num_dim, T, num_faces))
            for j in range(T):
                if os.path.isfile(landmark_path + str(j + 1) + '.npy'):
                    tmp = np.load(landmark_path + str(j + 1) + '.npy')
                    sample_numpy[:, :, j, 0] = tmp
                    dif = num_frames - T
                    num_tiles = int(dif / T)
                    if dif == 0:
                        sample_new = sample_numpy
                    elif num_tiles > 0 and dif > 0:
                        sample_numpy = torch.from_numpy(sample_numpy)
                        sample_numpy = tile(sample_numpy, 2, num_tiles)
                        sample_numpy = sample_numpy.numpy()
                        T_new = sample_numpy.shape[2]
                        dif_new = num_frames - T_new
                        if dif_new > 0:
                            sample_new = np.zeros((num_landmarks, num_dim, num_frames, num_faces))
                            sample_new[:, :, :T_new, :] = sample_numpy[:, :, :, :]
                            for k in range(dif_new):
                                sample_new[:, :, T_new + k, :] = sample_numpy[:, :, -1, :]
                        else:
                            sample_new = sample_numpy
                    elif num_tiles == 0 and dif > 0:
                        sample_new = np.zeros((num_landmarks, num_dim, num_frames, num_faces))
                        sample_new[:, :, 0:T, :] = sample_numpy[:, :, :, :]
                        for k in range(dif):
                            sample_new[:, :, T + k, :] = sample_numpy[:, :, -1, :]
                    class_numpy[s, :, :, :, :] = sample_new
                    s = s + 1  # nonzero sample
    np.save(landmark_path + c + '.npy', class_numpy)
    return class_numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facial landmark extractor')
    parser.add_argument("-i", "--landmark_folder", required=True, default='./data/AFEW_landmarks/',
                        description="path to extracted landmarks")
    parser.add_argument("-o", "--data_folder", required=True, default='./data/AFEW_data/',
                        description="path to extracted landmarks")
    arg = vars(parser.parse_args())
    classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Disgust', 'Neutral']
    part = ['Train', 'Val']
    num_frames = 150
    num_landmarks = 68
    num_dim = 2
    num_faces = 1
    AFEW_Data = []
    for p in part:
        for c in classes:
            landmark_path = arg.landmark_folder + '/{}/{}/'.format(p, c)
            class_data = afew_data_gen(landmark_path, c, num_frames, num_landmarks, num_dim, num_faces)
            AFEW_Data.append(class_data)
        p_data = np.concatenate(AFEW_Data[0:6], axis=0)
        p_data = data_normalization(p_data)
        N, C, T, V, M = p_data.shape
        rnd = np.random.permutation(N)
        shuffled_p_data = p_data[rnd]
        np.save(arg.data_folder + p + '.npy', shuffled_p_data)
        lbl_name = []
        lbls = []
        for c in range(len(classes)):
            for i in range(AFEW_Data[c].shape[0]):
                lbls.append(c)
        labels = np.array(lbls)
        shuffled_labels = labels[rnd]
        for i in range(N):
            if shuffled_labels[i] == 0:
                lbl_name.append('Angry' + str(i))
            elif shuffled_labels[i] == 1:
                lbl_name.append('Fear' + str(i))
            elif shuffled_labels[i] == 2:
                lbl_name.append('Sad' + str(i))
            elif shuffled_labels[i] == 3:
                lbl_name.append('Happy' + str(i))
            elif shuffled_labels[i] == 4:
                lbl_name.append('Disgust' + str(i))
            elif shuffled_labels[i] == 5:
                lbl_name.append('Surprise' + str(i))
            elif shuffled_labels[i] == 6:
                lbl_name.append('Neutral' + str(i))
            shuffled_labels = shuffled_labels.tolist()
            p_labels = (lbl_name, shuffled_labels)

            with open(arg.data_folder + p+'_labels.pkl', 'wb') as f:
                pickle.dump(p_labels, f)
