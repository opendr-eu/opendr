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
from numpy.lib.format import open_memmap
from scipy.spatial import Delaunay
import argparse


def find_graph_edges(x):
    points = np.transpose(x[0, :, 0, :, 0])
    print(points.shape)
    tri = Delaunay(points)
    neigh = tri.simplices
    print(neigh.shape)
    G = []
    N = neigh.shape[0]
    for i in range(N):
        G.append((neigh[i][0], neigh[i][1]))
        G.append((neigh[i][0], neigh[i][2]))
        G.append((neigh[i][1], neigh[i][2]))
    # connect the master node (nose) to all other nodes
    for i in range(51):
        G.append((i+1, 17))
    edges = G
    return edges


def gen_muscle_data(data, muscle_path):
    """Generate facial muscle data from facial landmarks"""
    N, C, T, V, M = data.shape
    edges = find_graph_edges(data)
    V_muscle = len(edges)
    fp_sp = open_memmap(muscle_path, dtype='float32', mode='w+', shape=(N, C, T, V_muscle, M))
    # Copy the landmark data to muscle placeholder tensor
    fp_sp[:, :, :, :V, :] = data
    for edge_id, (source_node, target_node) in enumerate(edges):
        fp_sp[:, :, :, edge_id, :] = data[:, :, :, source_node-1, :] - data[:, :, :, target_node-1, :]
    return fp_sp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facial muscle data generator.')
    parser.add_argument('--landmark_data_folder', default='./data/CASIA_10fold/')
    parser.add_argument('--muscle_data_folder', default='./data/muscle_data/')
    parser.add_argument('--dataset_name', default='CASIA')
    arg = parser.parse_args()
    part = ['Train', 'Val']
    for p in part:
        if arg.dataset_name == 'CASIA' or arg.dataset_name == 'CK+':
            for i in range(10):
                landmark_path = arg.landmark_data_folder + '/{}/{}_{}.npy'.format(arg.dataset_name, p, i)
                landmark_data = np.load(landmark_path)
                muscle_path = arg.muscle_data_folder + '/{}/{}_muscle_{}.npy'.format(arg.dataset_name, p, i)
                muscle_data = gen_muscle_data(landmark_data, muscle_path)
        elif arg.dataset_name == 'AFEW':
            landmark_path = arg.landmark_data_folder + '/{}/{}.npy'.format(arg.dataset_name, p)
            landmark_data = np.load(landmark_path)
            muscle_path = arg.muscle_data_folder + '/{}/{}_muscle.npy'.format(arg.dataset_name, p)
            muscle_data = gen_muscle_data(landmark_data, muscle_path)
