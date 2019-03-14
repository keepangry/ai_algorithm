#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-14 上午6:48
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : dataset.py
# @Software: PyCharm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def binary_iris(type=0, top_features=4, random_state=1):
    """

    :param type:  0: 0/1 , 1: -1/1
    :return:
    """
    iris = load_iris()
    X = iris.data[:, :top_features]
    y = iris.target
    # 转换为 1/-1 的二分类数据
    data_index = y != 0
    X = X[data_index]
    if type == 0:
        y = y[data_index] - 1
    else:
        y = (y[data_index]) * 2 - 3

    # split train test
    return train_test_split(
        X, y, test_size=0.25, random_state=random_state)


def multi_iris(top_features=4, random_state=1, onehot=False):
    """

    :param top_features:
    :param random_state:
    :return:
    """
    iris = load_iris()
    X = iris.data[:, :top_features]

    if onehot:
        y = iris.target.reshape(-1, 1)
        enc = OneHotEncoder(dtype=int, categories='auto')
        enc.fit(y)
        y = enc.transform(y).toarray()
    else:
        y = iris.target
    # split train test
    return train_test_split(
        X, y, test_size=0.25, random_state=random_state)


if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = binary_iris()
    X_train, X_test, y_train, y_test = multi_iris()
