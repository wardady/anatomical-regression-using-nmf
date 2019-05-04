#!/usr/bin/env python3
import numpy as np
import networkx as nx
from PIL import Image
from functools import reduce
from itertools import product


def get_image_line(img_name):
    return np.asarray(Image.open(img_name).convert('L').getdata(), np.float_)


def add_line(arr, line):
    return np.vstack((arr, line))


# weighting function - heat kernel
def f(t, y, i, j):
    # t depends on the range of the labels, so as the distance between data increases
    # in the dependant variable space, the weight decreases accordingly.
    return np.exp(-np.linalg.norm(y[:, i] - y[:, j]) / t)


# shortest path distance matrix
def d(v):
    # k-nearest neighbor graph with the Euclidean distance as edge weights is constructed
    graph = nx.Graph()
    graph.add_nodes_from(range(v.shape[1]))
    for i, j in product(range(v.shape[1]), range(v.shape[1])):
        graph.add_edge(i, j, weight=np.linalg.norm(v[:, i] - v[:, j]))
    # shortest path distance matrix D on the graph is calculated using Floyd's algorithm
    distance_matrix = nx.algorithms.shortest_paths.floyd_warshall_numpy(graph)
    for i in range(v.shape[1]):
        distance_matrix[i, i] = np.nan
    return distance_matrix


def m(d, v, f, t, y, i, j):
    if i == j:
        raise ValueError
    return np.nanmin(d(v)[i, j]) / np.linalg.norm(v[:, i] - v[:, j]) * f(t, y, i, j)


# constraints to minimize the divergence


def S_Rm(m, d, v, f, t, y, h):
    # in order to produce regressing features, we want the data to be smoothly separated
    upper_sum, lower_sum = np.float_(0), np.float_(0)
    for i, j in product(range(h.shape[1]), range(h.shape[1])):
        if i != j:
            upper_sum += m(d, v, f, t, y, i, j) * np.linalg.norm(h[:, i] - h[:, j])
            lower_sum += m(d, v, f, t, y, i, j)
    return upper_sum / lower_sum


# gradient based smoothing constraint
def S_G(w):
    # minimize the energy of the gradient of each of the basis W_j in the image coordinates
    pass


# this constraint attempts to minimize the number of basis components required to represent V
def S_O(w):
    return np.sum(w.T * w)


def divergence(alpha, beta, gamma, S_Rm, m, d, v, f, t, y, h, S_G, w, S_O):
    s = alpha * S_Rm(m, d, v, f, t, y, h) + beta * S_G(w) + gamma * S_O(w)
    for i, j in product(range(v.shape[0]), range(v.shape[1])):
        s += v[i, j] * np.log(v[i, j] / y[i, j]) - v[i, j] + y[i, j]
    return s


# update rules


def b():
    pass


def update_h():
    pass


def update_w():
    pass


if __name__ == '__main__':
    matrix = reduce(add_line, [get_image_line(f'testdata/img{i + 1}.png') for i in range(9)]).transpose()
