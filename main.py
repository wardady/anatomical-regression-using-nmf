#!/usr/bin/env python3
import numpy as np
from PIL import Image
from functools import reduce
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression


def get_image_line(img_name):
    return np.asarray(Image.open(img_name).convert('L').getdata(), np.float_)


def add_line(arr, line):
    return np.vstack((arr, line))


if __name__ == '__main__':
    matrix = reduce(add_line, [get_image_line(f'dataset/{i + 1}.png') for i in range(40)]).transpose()
    # get data
    model = NMF(2)
    w = model.fit_transform(matrix)
    h = model.components_
    y = np.matmul(w, h)
    # non negative matrix factorization
    age_array = np.empty(40, np.float_)
    with open('dataset/dataset.csv') as file:
        for i, line in enumerate(file):
            age_array[i] = np.float_(line.split()[1])
    # get ages
    reg = LinearRegression().fit(h.transpose()[:30], age_array[:30])
    print(reg.score(h.transpose()[30:], age_array[30:]))
    # get score
