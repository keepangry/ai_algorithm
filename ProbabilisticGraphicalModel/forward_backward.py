#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-13 下午2:15
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : forward_backward.py
# @Software: PyCharm

import numpy as np



if __name__ == "__main__":
    O = ['sunny', 'cloud', 'sunny', 'cloud', 'rain']
    O1 = ['sunny', 'cloud', 'sunny']

    # model
    states = ['sunny', 'cloud', 'rain']
    state2index = dict(zip(states, range(len(states))))
    trans_matrix = [
        [0.5, 0.3, 0.2],
        [0.6, 0.2, 0.2],
        [0.4, 0.2, 0.4],
    ]

    # p(O|model)
    p = 1
    z = 0
    for i in range(1, len(O1)):
        p *= trans_matrix[state2index[O[i-1]]][state2index[O[i]]]

    print(p)









