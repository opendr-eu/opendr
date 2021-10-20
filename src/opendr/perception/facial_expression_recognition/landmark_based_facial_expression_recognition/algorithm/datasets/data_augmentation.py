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

import numpy as np
import pickle
import torch
import argparse


def gen_aug_data(landmark_path, label_path, aug_path):
    data = np.load(landmark_path)
    data_aug = data
    with open(label_path, 'rb') as f:
        (lbl, lbl_name) = pickle.load(f)
    for i in range(3):
        # noise
        noise = np.random.normal(0, 10, data.shape)
        noised_data = data + noise
        # rotate
        noised_data = torch.from_numpy(noised_data)
        N, C, T, V, M = noised_data.size()
        noised_data = noised_data.permute(0, 4, 2, 3, 1).contiguous().view(N, M, T, V, C)
        theta = np.random.uniform(-np.pi / 10, np.pi / 10)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        R = torch.from_numpy(R)
        rotated_data = torch.matmul(noised_data, R)
        rotated_data = (rotated_data.permute(0, 4, 2, 3, 1).contiguous().view(N, C, T, V, M)).numpy()
        noised_data = (noised_data.permute(0, 4, 2, 3, 1).contiguous().view(N, C, T, V, M)).numpy()
        # flip
        data_noised_flipped = np.flip(noised_data, 2)
        data_rotated_flipped = np.flip(rotated_data, 2)
        data_aug = np.concatenate((data_aug, noised_data, rotated_data, data_noised_flipped,
                                   data_rotated_flipped), axis=0)
    data_flipped = np.flip(data, 2)
    aug_data = np.concatenate((data_aug, data_flipped))
    N, C, T, V, M = aug_data.shape
    p = np.random.permutation(N)
    shuffled_data_aug = aug_data[p]
    np.save(aug_path+'shuffled_data_aug.npy', shuffled_data_aug)
    lbls_aug = []
    lbl_name_aug = []
    for i in range(14):
        lbls_aug = lbls_aug + lbl
        lbl_name_aug = lbl_name_aug + lbl_name
    lbls_aug = np.array(lbls_aug)
    lbls_aug = (lbls_aug[p]).tolist()
    lbl_name_aug = np.array(lbl_name_aug)
    lbl_name_aug = lbl_name_aug[p].tolist()
    lbls = (lbl_name_aug, lbls_aug)
    with open(aug_path+'labels_aug.pkl', 'wb') as f:
        pickle.dump(lbls, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data augmentation for AFEW dataset')
    parser.add_argument('--data_folder', default='./data/AFEW_data/')
    parser.add_argument('--aug_data_folder', default='./data/AFEW_aug_data/')
    arg = parser.parse_args()
    part = ['Train']
    for p in part:
        landmark_path = arg.data_folder + '{}.npy'.format(p)
        label_path = arg.data_folder + '{}_label.pkl'.format(p)
        aug_path = arg.aug_data_folder + '{}_aug.npy'.format(p)
        gen_aug_data(landmark_path, label_path, aug_path)
