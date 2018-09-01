# encoding: utf-8
'''
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: entropy.py
@time: 18-8-25 下午10:03
@desc:
'''
import numpy as np
from collections import Counter
from math import log2, log


def discrete_entropy(vector):
    """
    越混乱，熵越大。
    二值熵在 0-1 之间
    多值熵可以超过 1 !!

    :param vector: np.array
    :return:
    """
    length = vector.shape[0]
    if length == 0:
        raise("向量长度为0")
    counter = Counter(vector)

    entropy = 0
    for value in counter.values():
        P_i = value / length
        entropy -= P_i * log2(P_i)
    return entropy


def gini(vector):
    """
    最大值为1，表示无序，均匀。
    :param vector:
    :return:
    """
    length = vector.shape[0]
    if length == 0:
        raise("向量长度为0")
    counter = Counter(vector)

    gini = 1
    for value in counter.values():
        P_i = value / length
        gini -= P_i**2
    return gini


if __name__ == "__main__":
    # print(discrete_entropy(np.array([1, 1, 1, 1])))
    # print(discrete_entropy(np.array([1, 1, 1, 0])))
    print(discrete_entropy(np.array([1, 2, 3, 3, 2, 4, 6, 6])))
    print(gini(np.array([1, 2, 3, 3, 2, 4, 6, 6])))





