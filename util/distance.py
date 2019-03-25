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


def cos_sim(vec, vecs):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """

    dot_product = np.dot(vecs, vec)
    norm_a = np.linalg.norm(vec)
    norm_b = np.linalg.norm(vecs, axis=1)
    return dot_product / (norm_a * norm_b)
