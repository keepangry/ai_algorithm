# encoding: utf-8
'''
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: loss.py
@time: 18-8-28 下午9:45
@desc:
'''

import numpy as np


def root_mean_square(vector1, vector2):
    return np.sqrt(np.mean(np.square(vector1 - vector2)))


if __name__ == "__main__":

    print(np.var(np.array([1, 2, 3])))
