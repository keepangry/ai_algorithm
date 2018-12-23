# encoding: utf-8
'''
@author: yangsen
@license:
@contact:
@software:
@file: numpy_mat.py
@time: 18-8-25 下午9:56
@desc:
'''
import numpy as np
a = np.arange(9).reshape(3,3)


# 行
a[1]
a[[1,2]]
a[np.array([1,2])]

# 列
a[:,1]
a[:,[1,2]]
a[:,np.array([1,2])]