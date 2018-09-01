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

def mean_square(vector):
    return np.square(vector - vector.mean()).sum()


if __name__ == "__main__":
    a = np.array([1,2,3])
    print(mean_square(a))