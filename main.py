#!/usr/bin/env python3
import numpy as np
import networkx as nx
from PIL import Image
from functools import reduce
from itertools import product
from sklearn.decomposition import NMF


def get_image_line(img_name):
    return np.asarray(Image.open(img_name).convert('L').getdata(), np.float_)


def add_line(arr, line):
    return np.vstack((arr, line))


# weighting function - heat kernel
def function_f(t, y, i, j):
    # t depends on the range of the labels, so as the distance between data increases
    # in the dependant variable space, the weight decreases accordingly.
    return np.exp(-np.linalg.norm(y[:, i] - y[:, j]) / t)


# shortest path distance matrix
def matrix_d(v):
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


def calculate_m(matrix_d, v, function_f, t, y, i, j):
    return np.nanmin(matrix_d(v)[i, j]) / np.linalg.norm(v[:, i] - v[:, j]) * function_f(t, y, i, j) if i != j else \
        function_f(t, y, i, j)


# constraints to minimize the divergence


def calculate_S_Rm(calculate_m, matrix_d, v, function_f, t, y, h):
    # in order to produce regressing features, we want the data to be smoothly separated
    upper_sum, lower_sum = np.float_(0), np.float_(0)
    for i, j in product(range(h.shape[1]), range(h.shape[1])):
        if i != j:
            m = calculate_m(matrix_d, v, function_f, t, y, i, j)
            upper_sum += m * np.linalg.norm(h[:, i] - h[:, j])
            lower_sum += m
    return upper_sum / lower_sum


# gradient based smoothing constraint
def calculate_S_G(w):
    # minimize the energy of the gradient of each of the basis W_j in the image coordinates
    return sum(np.trapz(np.square(np.gradient(w[:, j]))) for j in range(w.shape[1]))


# this constraint attempts to minimize the number of basis components required to represent V
def calculate_S_O(w):
    return np.sum(np.matmul(w.T, w))


def divergence(alpha, beta, gamma, calculate_S_Rm, calculate_m, matrix_d, v, function_f, t, y, h, calculate_S_G, w, calculate_S_O):
    return sum(v[i, j] * np.log(v[i, j] / y[i, j]) - v[i, j] + y[i, j]
               for i, j in product(range(v.shape[0]), range(v.shape[1]))) \
           + alpha * calculate_S_Rm(calculate_m, matrix_d, v, function_f, t, y, h) + beta * calculate_S_G(w) + gamma * calculate_S_O(w)


# update rules


def calculate_b(alpha, calculate_m, matrix_d, v, function_f, t, y, h, k, l):
    lower_sum = sum(calculate_m(matrix_d, v, function_f, t, y, i, j) for i, j in product(range(v.shape[0]), range(v.shape[1])))
    upper_sum = sum(h[k, xk] * calculate_m(matrix_d, v, function_f, t, y, xk, l) for xk in range(h.shape[1]))
    return 1 - 4 * alpha * upper_sum / lower_sum


def update_h(calculate_b, alpha, calculate_m, matrix_d, v, function_f, t, y, h, k, l, w):
    b = calculate_b(alpha, calculate_m, matrix_d, v, function_f, t, y, h, k, l)
    big_sum = sum(v[i, l] * w[i, k] * h[k, l] / np.sum(w[i, xk] * h[xk, l] for xk in range(w.shape[1])) for i in range(v.shape[0]))
    m_sum = sum(calculate_m(matrix_d, v, function_f, t, y, i, j) for i, j in product(range(v.shape[0]), range(v.shape[1])))
    m = calculate_m(matrix_d, v, function_f, t, y, k, l)
    return (-b + np.sqrt(np.square(b) + big_sum * 16 * alpha * m / m_sum)) / (8 * alpha * m / m_sum)


def matrix_g(w):
    pass


def update_w_1(w, v, h, beta, ):
    pass


def update_w_2():
    pass


if __name__ == '__main__':
    matrix = reduce(add_line, [get_image_line(f'testdata/img{i + 1}.png') for i in range(9)]).transpose()
    for i, j in product(range(matrix.shape[0]), range(matrix.shape[1])):
        matrix[i, j] += 1
    model = NMF(2)
    w = model.fit_transform(matrix)
    h = model.components_
    y = np.matmul(w, h)
    print(divergence(1, 1, 1, calculate_S_Rm, calculate_m, matrix_d, matrix, function_f, 1, y, h, calculate_S_G, w, calculate_S_O))
