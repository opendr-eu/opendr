"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import numpy as np

num_node = 25
self_link = [(i, i) for i in range(num_node)]
in_edge_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
in_edge = [(i - 1, j - 1) for (i, j) in in_edge_ori_index]
out_edge = [(j, i) for (i, j) in in_edge]
neighbor = in_edge + out_edge


def get_hop(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, in_edge, out_edge):
    I = get_hop(self_link, num_node)
    In = normalize_digraph(get_hop(in_edge, num_node))
    Out = normalize_digraph(get_hop(out_edge, num_node))
    A = np.stack((I, In, Out))
    return A


class NTUGraph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.in_edge = in_edge
        self.out_edge = out_edge
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, in_edge, out_edge)
        else:
            raise ValueError()
        return A
