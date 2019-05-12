#!/usr/bin/env python3
import numpy as np
from PIL import Image
from sklearn.decomposition import NMF
from functools import reduce


def get_image_line(img_name):
    return np.asarray(Image.open(img_name).convert('L').getdata(), np.float_)


def add_line(arr, line):
    return np.vstack((arr, line))


if __name__ == '__main__':
    matrix = reduce(add_line, [get_image_line(f'testdata/img{i + 1}.png') for i in range(9)]).transpose()
    model = NMF(2)
    w = model.fit_transform(matrix)
    h = model.components_
    y = np.matmul(w, h)
