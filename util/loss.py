# encoding: utf-8
'''
@author: yangsen
@license:
@contact: 0@keepangry.com
@software:
@file: loss.py
@time: 18-8-28 下午9:45
@desc:
'''

import numpy as np


def root_mean_square(vector1, vector2):
    return np.sqrt(np.mean(np.square(vector1 - vector2)))


if __name__ == "__main__":

    print(np.var(np.array([1, 2, 3])))
