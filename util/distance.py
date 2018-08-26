# encoding: utf-8
'''
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: distance.py
@time: 18-8-23 下午9:07
@desc:
'''

import numpy as np


def euclidean(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)