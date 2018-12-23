# encoding: utf-8
'''
@author: yangsen
@license:
@contact: 0@keepangry.com
@software:
@file: distance.py
@time: 18-8-23 下午9:07
@desc:
'''

import numpy as np


def euclidean(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)