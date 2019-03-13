#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-14 上午6:42
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : LogisticRegression.py
# @Software: PyCharm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import classification_report

from util.dataset import binary_iris


class LogisticRegression(object):
    def __init__(self, max_iter=1000, loss='square_loss', learning_rate=0.001, min_iter_loss=0.01,
                 optimize_method='stochastic_gradient_descent', batch_num=0.1, random_state=0, alpha=1.0, regularizer=None):
        """

        :param method:
        :param max_iter:
        :param loss: (square_loss, abs_loss, ridge, lasso) ridge: square_loss + L2 , lasso square_loss + L1
        :param learning_rate:
        :param min_iter_loss:
        :param optimize_method: (gradient_descent, stochastic_gradient_descent, batch_gradient_descent)
        :param batch_num: (0,1) 代表比例
        :param random_state:
        :param alpha: when loss in (ridge, lasso) alpha代表正则项系数
        """

        self.w = None

        self.max_iter = max_iter
        """
        # 最开始learning_rate设置为0.01，损失越来越大，很是纳闷，不知道哪里出问题。
        最后发现是learning_rate 0.01过大导致的，因为如果是全体样本进行计算，梯度过大，容易跑飞，设置较小的learning_rate便正常了。

        使用随机梯度下降，设置过小的learning_rate容易优化过慢
        """
        self.learning_rate = learning_rate
        self.optimize_method = optimize_method
        self.random_state = random_state
        self.min_iter_loss = min_iter_loss
        self.batch_num = batch_num
        self.loss = loss
        self.alpha = alpha

        self.regularizer = regularizer

        # 这一项不能设置过大，会出现问题。
        self.regularizer_weight = 0.1

    def fit(self, X, y):
        self.train_num = X.shape[0]
        self.feature_num = X.shape[1]

        self.fit_by_gradient_descent(X, y)

    def gradient_descent_sample_choice(self, X, y):
        size = int(self.batch_num*self.train_num)
        choice_indexes = np.random.choice(self.train_num, replace=False, size=size)
        curr_X = X[choice_indexes]
        curr_y = y[choice_indexes]
        return curr_X, curr_y

    def gradient(self, theta, curr_hx, curr_X, curr_y):
        pass

    def h(self, W, X):
        return 1 / (1 + np.exp(-W.dot(X.T)))

    def log_loss(self, W, X, y):
        h = self.h(W, X)
        # 对数似然是负值，需要最大化。 取其负数，为 负对数似然损失。进行最小化
        r = 0
        if self.regularizer == 'l2':
            r = self.regularizer_weight * np.sum(W**2)
        return (-np.sum(y*np.log(h) + (1-y)*np.log(1-h)) + r) / X.shape[0]

    def fit_by_gradient_descent(self, X, y):
        """
        梯度下降求解
        :param X:
        :param y:
        :return:
        """
        # 迭代次数
        b = np.ones_like(X[:, [0]])
        X = np.hstack([X, b])

        # 初始化参数
        np.random.seed(self.random_state) # 注意此处只有一次有效
        # W包含b
        W = np.random.random(X.shape[1])

        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1

            # 选择当前进行梯度下降优化的样本
            curr_X, curr_y = self.gradient_descent_sample_choice(X, y)
            sample_num = curr_X.shape[0]
            h = self.h(W, curr_X)
            W -= self.learning_rate * 1 / sample_num * ((h - curr_y).dot(curr_X) + self.regularizer_weight*W)

            # 计算loss
            loss = self.log_loss(W, X, y)
            print("loss: %s" % loss)

        self.w = W

    def predict(self, X):
        return np.round(self.predict_prob(X)).astype(int)

    def predict_prob(self, X):
        b = np.ones_like(X[:, [0]])
        X = np.hstack([X, b])
        return self.h(self.w, X)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = binary_iris(top_features=3, random_state=1)
    clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs')
    clf.fit(X_train, y_train)
    print("sklearn LR: %s" % clf.score(X_test, y_test))

    # lr = LogisticRegression(learning_rate=0.05, max_iter=1000)
    # lr.fit(X_train, y_train)
    # y_pred = lr.predict(X_test)
    # print(classification_report(y_test, y_pred, target_names=['label:0', 'label:1']))

    lr = LogisticRegression(learning_rate=0.05, max_iter=1000, regularizer='l2')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['label:0', 'label:1']))
