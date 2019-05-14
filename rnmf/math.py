import numpy as np
import networkx as nx
from itertools import product
from scipy.ndimage.filters import laplace
from sklearn.decomposition import NMF


def nested_range(tupl):
    return product(range(tupl[0]), range(tupl[1]))


# weighting function
def heat_kernel(y, t, i, j):
    # t depends on the range of the labels, so as the distance between data increases
    # in the dependant variable space, the weight decreases accordingly
    return np.exp(-np.linalg.norm(y[:, i] - y[:, j]) / t)


def shortest_path_distance_matrix(v):
    # k-nearest neighbor graph with the Euclidean distance as edge weights is constructed
    graph = nx.Graph()
    graph.add_nodes_from(range(v.shape[1]))
    for i, j in product(range(v.shape[1]), range(v.shape[1])):
        graph.add_edge(i, j, weight=np.linalg.norm(v[:, i] - v[:, j]) if i != j else np.nan)
    # shortest path distance matrix D on the graph is calculated using Floyd's algorithm
    return nx.algorithms.shortest_paths.floyd_warshall_numpy(graph)


def matrix_m(distance_matrix, v, y, t):
    # weights are inversely proportional to the geodesic distance between the sample points, thus penalizing outliers
    m = np.empty(v.shape, np.float_)
    for i, j in nested_range(v.shape):
        m[i, j] = np.nanmin(distance_matrix[i, j]) / np.linalg.norm(v[:, i] - v[:, j]) * heat_kernel(y, t, i, j) \
            if i != j else heat_kernel(y, t, i, j)
    return m


# constraints to minimize the cost function (divergence)


# incorporating the regression methodology by imposing the following constraint
def calculate_S_Rm(matrix_m, h):
    # in order to produce regressing features, we want the data to be smoothly separated
    numer = sum(matrix_m[i, j] * np.linalg.norm(h[:, i] - h[:, j]) for i, j in nested_range(matrix_m.shape))
    denom = sum(matrix_m[i, j] for i, j in nested_range(matrix_m.shape))
    return numer / denom


# gradient based smoothing constraint
def calculate_S_G(w):
    # minimize the energy of the gradient of each of the basis W_j in the image coordinates
    return np.trapz(np.square(np.gradient(w, axis=0))).sum()


# this constraint attempts to minimize the number of basis components required to represent V
def calculate_S_O(w):
    return np.matmul(w.T, w).sum()


# the inclusion of the above three constraints leads to the following constrained divergence to be minimized
# the constants α, β and γ are empirically determined such that we get non-negative updates
def divergence(alpha, beta, gamma, matrix_m, v, y, w, h):
    div = alpha * calculate_S_Rm(matrix_m, h) + beta * calculate_S_G(w) + gamma * calculate_S_O(w)
    for i, j in nested_range(v.shape):
        div -= v[i, j]
        if v[i, j] and y[i, j]:
            div += v[i, j] * np.log(v[i, j] / y[i, j]) + y[i, j]
    return np.min(div)


# update rules found by minimization of the above cost function


def calculate_b(alpha, matrix_m, h, k, l):
    return 1 - 4 * alpha * sum(h[k, xk] * matrix_m[xk, l] for xk in range(h.shape[1])) / matrix_m.sum()


def update_h(alpha, matrix_m, v, w, h, k, l):
    b = calculate_b(alpha, matrix_m, h, k, l)
    big_sum = np.float_(0)
    for i in range(v.shape[0]):
        small_sum = sum(w[i, xk] * h[xk, l] for xk in range(w.shape[1]))
        if small_sum:
            big_sum += v[i, l] * w[i, k] * h[k, l] / small_sum
    val = 8 * alpha * matrix_m[k, l] / matrix_m.sum()
    return (-b + np.sqrt(np.square(b) + big_sum * 2 * val)) / val


def matrix_g(w):
    return -laplace(w)


def update_w_1(beta, gamma, g, v, w, h, k, l):
    numer = np.float_(0)
    for j in range(v.shape[1]):
        small_sum = sum(w[xk, l] * h[l, j] for xk in range(w.shape[0]))
        if small_sum:
            numer += v[k, j] * h[l, j] / small_sum
    denom = h[l, :].sum() + beta * g[k, l] + gamma * w[k, :].sum()
    return numer / denom


def update_w_2(w, k, l):
    column_sum = w[:, l].sum()
    return w[k, l] / column_sum if column_sum else 0


def minimize(v, difference):
    model = NMF(2)
    w = model.fit_transform(v)
    h = model.components_
    y = np.matmul(w, h)
    distance_matrix = shortest_path_distance_matrix(v)
    m = matrix_m(distance_matrix, v, y, 1)
    prev_div, new_div = np.float_(0), divergence(1, 1, 1, m, v, y, w, h)
    while np.abs(new_div - prev_div) > difference:
        # update h
        new_h = np.empty(h.shape, np.float_)
        for k, l in nested_range(new_h.shape):
            new_h[k, l] = update_h(1, m, v, w, h, k, l)
        h = new_h
        # update w
        new_w = np.empty(w.shape, np.float_)
        g = matrix_g(w)
        for k, l in nested_range(new_w.shape):
            new_w[k, l] = update_w_1(1, 1, g, v, w, h, k, l)
        for k, l in nested_range(w.shape):
            w[k, l] = update_w_2(new_w, k, l)
        # update variables
        y = np.matmul(w, h)
        distance_matrix = shortest_path_distance_matrix(v)
        m = matrix_m(distance_matrix, v, y, 1)
        prev_div, new_div = prev_div, divergence(1, 1, 1, m, v, y, w, h)
