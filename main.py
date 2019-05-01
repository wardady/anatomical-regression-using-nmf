#!/usr/bin/env python3
import numpy as np
from PIL import Image
from functools import reduce


def get_image_line(img_name):
    return np.asarray(Image.open(img_name).convert('L').getdata(), np.uint8)


def add_line_to_array(arr, line):
    return np.vstack((arr, line))


if __name__ == '__main__':
    array = reduce(add_line_to_array, [get_image_line(f'testdata/img{i+1}.png') for i in range(9)])
