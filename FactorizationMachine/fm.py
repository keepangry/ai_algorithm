#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  : yangsen
# @Site    : 
# @File    : fm.py
# @Software: PyCharm
"""
https://blog.csdn.net/lieyingkub99/article/details/80897743
https://www.cnblogs.com/pinard/p/6370127.html
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from util.value_function import sigmoid


class FactorizationMachine(object):
    def __init__(self):
        self.k = 2
        self.learning_rate = 0.1
        self.max_iter = 1000

    def fit(self, instances, labels):
        self.instances = instances
        self.labels = labels
        self.instances_num = instances.shape[0]
        self.feature_dim = instances.shape[1]

        random_generator = np.random.RandomState(1)
        # 初始化参数
        # n * k
        self.V = random_generator.rand(self.feature_dim, self.k)
        self.W = random_generator.rand(self.feature_dim)
        self.b = random_generator.rand()

        self.gradient_descent()

    def instances_batch(self, batch_num):
        if 0 < batch_num < 1:
            # 无放回
            choice_indexes = np.random.choice(self.instances_num, replace=False, size=int(batch_num*self.instances_num))
        elif 1 <= batch_num < self.instances_num:
            choice_indexes = np.random.choice(self.instances_num, replace=False, size=batch_num)
        else:
            raise Exception("batch_num 参数错误.")
        return self.instances[[choice_indexes]], self.labels[choice_indexes]

    def get_hypothesis_value(self, X):
        """
        hypothesis function value
        :return:
        """
        W = self.W
        V = self.V
        b = self.b

        linear_term = X.dot(W)
        # https://www.cnblogs.com/pinard/p/6370127.html  2.11
        quadratic_term = np.sum(np.square(V.T.dot(X.T)), axis=0) - \
                         np.sum(np.sum(np.square(V.T * np.stack((X, X), axis=1)), axis=1), axis=1)

        return self.b + linear_term + quadratic_term


    def gradient_descent(self):
        pre_loss = 999999
        batch_size = 10
        W = self.W
        V = self.V
        b = self.b

        for i in range(self.max_iter):
            curr_X, curr_y = self.instances_batch(batch_num=1)
            curr_X, curr_y = curr_X[0], curr_y[0]
            # f(x) 预测值
            curr_h = sigmoid(self.get_hypothesis_value(curr_X))
            loss = - np.mean(curr_y * np.log(curr_h) + (1-curr_y) * np.log(1-curr_h))
            print(loss)
            loss_derivative = curr_h - curr_y
            print(curr_X)
            linear_term = np.sum(curr_X * W)
            # quadratic_term

            pre = (sigmoid(curr_y * self.get_hypothesis_value(curr_X)) - 1) * curr_y

            # new_b = self.b - self.learning_rate * np.mean(loss_derivative)
            # new_W = self.W - self.learning_rate * np.mean(self.W.reshape(-1, 1) * loss_derivative, axis=1)
            # new_V = self.V - self.learning_rate * np.mean(self.V.reshape(-1, 1) * loss_derivative, axis=1).reshape(self.feature_dim, self.k)
            # self.b, self.W, self.V = new_b, new_W, new_V
            new_b = self.b - self.learning_rate * np.mean(pre * 1)
            new_W = self.W - self.learning_rate * pre.dot(curr_X) / batch_size
            # 减去对角线值 TODO: 写不出来，特别还是batch的
            # new_V = self.V - self.learning_rate * ( self.V.T.dot(curr_X.T) -  ).dot(curr_X)





    # def gradient_descent(self):
    #     pre_loss = 999999
    #     batch_size = 10
    #
    #     for i in range(self.max_iter):
    #         curr_X, curr_y = self.instances_batch(batch_num=batch_size)
    #
    #         # f(x) 预测值
    #         curr_h = sigmoid(self.get_hypothesis_value(curr_X))
    #         loss = - np.mean(curr_y * np.log(curr_h) + (1-curr_y) * np.log(1-curr_h))
    #         print(loss)
    #         loss_derivative = curr_h - curr_y
    #         print(curr_X)
    #         # https://www.cnblogs.com/pinard/p/6370127.html  4.3  loss^C 前半部分
    #         pre = (sigmoid(curr_y * self.get_hypothesis_value(curr_X)) - 1) * curr_y
    #
    #         # new_b = self.b - self.learning_rate * np.mean(loss_derivative)
    #         # new_W = self.W - self.learning_rate * np.mean(self.W.reshape(-1, 1) * loss_derivative, axis=1)
    #         # new_V = self.V - self.learning_rate * np.mean(self.V.reshape(-1, 1) * loss_derivative, axis=1).reshape(self.feature_dim, self.k)
    #         # self.b, self.W, self.V = new_b, new_W, new_V
    #         new_b = self.b - self.learning_rate * np.mean(pre * 1)
    #         new_W = self.W - self.learning_rate * pre.dot(curr_X) / batch_size
    #         # 减去对角线值 TODO: 写不出来，特别还是batch的
    #         # new_V = self.V - self.learning_rate * ( self.V.T.dot(curr_X.T) -  ).dot(curr_X)


def iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    # 转换为 1/-1 的二分类数据
    data_index = y != 2
    X = X[data_index]
    y = y[data_index]
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# def cancer():
#     breast_cancer = load_breast_cancer()
#     X = breast_cancer.data
#     y = (breast_cancer.target * 2) - 1
#     return train_test_split(
#         X, y, test_size=0.33, random_state=3)


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = iris()

    fm = FactorizationMachine()
    fm.fit(X_train, y_train)

    # adaboost.fit(np.array([0,1,2,3,4,5,6,7,8,9])[::-1].reshape((10, 1)),
    #              np.array([1,1,1,-1,-1,-1,1,1,1,-1])[::-1])
    # adaboost.fit(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((10, 1)),
    #              np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1]))


    # from sklearn.linear_model import LogisticRegression
    # clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, (y_train+1)/2)
    # # lr_pred = clf.predict(X_test)
    # print("LR: %s" % clf.score(X_test, (y_test+1)/2))
    #
    # from sklearn.ensemble import AdaBoostClassifier
    # from sklearn.tree import DecisionTreeClassifier
    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
    # bdt.fit(X_train, y_train)
    # print("sklearn ada: %s" % bdt.score(X_test, y_test))
