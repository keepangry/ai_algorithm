#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  : yangsen
# @Site    : 
# @File    : value_function.py.py
# @Software: PyCharm

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

