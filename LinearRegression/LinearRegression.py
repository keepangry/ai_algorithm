# encoding: utf-8
'''
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: 0@keepangry.com
@software: garner
@file: LinearRegression.py
@time: 18-9-1 下午10:41
@desc:
'''

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model



class LinearRegression(object):
    def __init__(self, method="gradient_descent", max_iter=10000, alpha=0.001, min_iter_loss=0.01):
        self.w = None
        self.method = method

        self.max_iter = max_iter
        """
        # 最开始alpha设置为0.01，损失越来越大，很是纳闷，不知道哪里出问题。
        最后发现是alpha 0.01过大导致的，因为如果是全体样本进行计算，梯度过大，容易跑飞，设置较小的alpha便正常了。

        使用随机梯度下降，设置过小的alpha容易优化过慢
        """
        self.alpha = alpha
        self.min_iter_loss = min_iter_loss


    def fit(self, X, y):
        if self.method == 'least_square':
            self.fit_by_least_square(X, y)
        elif self.method == 'gradient_descent':
            self.fit_by_gradient_descent(X, y)


    def fit_by_gradient_descent(self, X, y):
        """
        梯度下降求解
        :param X:
        :param y:
        :return:
        """
        # 迭代次数
        b = np.ones_like(X[:,[0]])
        X = np.hstack([X, b])

        # 初始化参数
        np.random.seed(0) # 注意此处只有一次有效
        theta = np.random.random(X.shape[1])
        pre_loss = 9999999999

        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1

            # TODO: 全样本梯度下降、随机梯度下降、批梯度下降
            # 求当前预测值
            hx = X.dot(theta)

            # 求梯度
            # TODO: 抽象出来，根据损失函数进行计算梯度
            gradient = (hx - y).dot(X)

            # 更新参数
            theta -= self.alpha * gradient

            # 判断是否收敛，方法1：参数的变化是否以及过小
            # 方法2： 损失是否已经不在降低
            # 计算平方损失，即二阶范数的平方
            curr_loss = sum((X.dot(theta) - y)**2)
            print("iter_num: {}, square loss: {}".format(iter_num, curr_loss))
            if pre_loss - curr_loss < self.min_iter_loss:
                break
            else:
                pre_loss = curr_loss
        self.w = theta


    def fit_by_least_square(self, X, y):
        """
        最小二乘求解
        X = (X,b)
        W = (X^T X)^-1 X^T Y

        :param X:
        :param y:
        :return:
        """
        b = np.ones_like(X[:,[0]])
        X = np.hstack([X, b])
        # 求
        self.w = np.linalg.inv( X.T.dot(X) ).dot(X.T).dot(y)


    def predict(self, X):
        b = np.ones_like(X[:,[0]])
        X = np.hstack([X, b])
        return X.dot(self.w)




if __name__ == "__main__":
    diabetes = datasets.load_diabetes()
    # 选1个特征，目的用于可视化
    X = diabetes.data[:,[8]]
    y = diabetes.target

    # 人造X
    # X = np.arange(0., 10., 0.2).reshape(50,1)
    # y = ( 2 * X + 5 + np.random.randn(1) ).reshape(50,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # lr = LinearRegression('least_square')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    plt.scatter(X_test, y_test, color='black')  # 散点输出
    plt.plot(X_test[:, 0], y_pred, color='blue', linewidth=2)  # 预测输出
    plt.show()