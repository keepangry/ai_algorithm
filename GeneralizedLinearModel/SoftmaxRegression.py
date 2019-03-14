#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-14 上午7:56
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : SoftmaxRegression.py
# @Software: PyCharm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

from util.dataset import multi_iris


class SoftmaxRegression(object):
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

        # 对y进行onehot
        y = y.reshape(-1, 1)
        enc = OneHotEncoder(dtype=int, categories='auto')
        enc.fit(y)
        y = enc.transform(y).toarray()

        self.category_num = y[0].shape[0]

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
        # W : category_num * (feature_num+1)
        # X : sample_num * (feature_num+1)
        # W.dot(X.T) :  category_num * sample_num
        samples_category_exp = np.exp(W.dot(X.T))

        # samples_z 归一到概率项. sample_num
        samples_z = np.sum(samples_category_exp, axis=0)

        # category_num * sample_num
        h = samples_category_exp / samples_z
        return h.T

    def cross_entropy_loss(self, W, X, y):
        h = self.h(W, X)
        # 对数似然是负值，需要最大化。 取其负数，为 负对数似然损失。进行最小化
        # r = 0
        # if self.regularizer == 'l2':
        #     r = self.regularizer_weight * np.sum(W**2)

        return -np.sum(np.multiply(np.log(h), y)) / y.shape[0]

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
        np.random.seed(self.random_state)  # 注意此处只有一次有效
        # W包含b
        W = np.random.random((self.category_num, X.shape[1]))

        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1

            # 选择当前进行梯度下降优化的样本
            curr_X, curr_y = self.gradient_descent_sample_choice(X, y)
            sample_num = curr_X.shape[0]
            h = self.h(W, curr_X)

            r = 0
            if self.regularizer == 'l2':
                r = self.regularizer_weight * W
            # TODO：关键是对 softmax 交叉熵损失 更新公式的推到。更新公式本身很简单。
            # https://blog.csdn.net/tkyjqh/article/details/78367369
            W -= self.learning_rate * 1 / sample_num * ((h - curr_y).T.dot(curr_X) + r)

            # 计算loss
            loss = self.cross_entropy_loss(W, X, y)
            print("loss: %s" % loss)

        self.w = W

    def predict(self, X):
        return np.argmax(self.predict_prob(X),axis=1)

    def predict_prob(self, X):
        b = np.ones_like(X[:, [0]])
        X = np.hstack([X, b])
        return self.h(self.w, X)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = multi_iris(top_features=3, random_state=4, onehot=False)

    sr = SoftmaxRegression(learning_rate=0.03, max_iter=2000, regularizer='l2')
    sr.fit(X_train, y_train)
    y_pred = sr.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['label:0', 'label:1', 'label:2']))

    clf = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='auto')
    clf.fit(X_train, y_train)
    print("sklearn LR: %s" % clf.score(X_test, y_test))
