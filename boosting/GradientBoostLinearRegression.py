#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/3 5:02 PM
# @Author  : yangsen
# @Site    : 
# @File    : GradientBoostRegressionTree.py
# @Software: PyCharm
import numpy as np
from decision_tree.CARTRegression import CARTRegression
from GeneralizedLinearModel.LinearRegression import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import random
"""
提升LR
方式1、使用残差拟合。  第二轮预测基本就是一条直线了，残差直接将为0，无法优化

方式2、使用 y_train - 累积。 线性叠加还是线性，无法产生非线性

方式3、非线性叠加

结论：
线性基分类器，无法通过加法模型进行提升到非线性。



"""

class GBLR(object):

    def __init__(self, learning_rate=1.0, subsample=1, n_estimators=10, min_samples_split=3, max_depth=2):
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.cart_trees = []
        self.y_list = []

    def fit(self, X_train, y_train):
        X = X_train
        y = y_train
        sum_y = np.zeros(y_train.shape[0])
        for i in range(self.n_estimators):
            # cart = CARTRegression(max_depth=self.max_depth, leaf_min_samples=self.min_samples_split)
            cart = LinearRegression(optimize_method='batch_gradient_descent', max_iter=1000, loss='abs_loss', alpha=1,
                                    batch_num=0.2, learning_rate=0.01, min_iter_loss=0.01, random_state=i)
            cart.fit(X, y)
            self.cart_trees.append(cart)
            y_pred = cart.predict(X_train)
            """
                此处，尝试拟合y_train - 当前总 剩余的，而不是预测的。
            """

            print("Tree %s        Mean squared error: %.6f" % (i, mean_squared_error(y, y_pred)))
            # 当为平方损失时，即拟合残差(y-pred) /  负梯度-(pred-y)
            self.y_list.append(y)
            # y = -(y_pred - y)*self.learning_rate
            sum_y += y_pred
            y = (y_train - sum_y)*self.learning_rate



    def predict(self, X_test):
        y = np.zeros(X_test.shape[0])
        for cart in self.cart_trees:
            y += cart.predict(X_test)
        return y


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()

    # X_te    st = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    # y_test = np.sin(X_test).ravel()

    X_test = X
    y_test = y

    gblr = GBLR(learning_rate=1, n_estimators=10)
    gblr.fit(X, y)
    y_gblr_pred = gblr.predict(X_test)
    print("gblr         Mean squared error: %.6f" % (mean_squared_error(y_test, y_gblr_pred)))

    ## 排查为什么gb无效
    result = np.array([gblr.cart_trees[i].predict(X_test) for i in range(10)])

    """
    result.sum(axis=1)
    array([ 1.04704103e+02, -8.07457930e-03,  3.10175660e-01,  3.06333904e-01,
        3.06380280e-01,  3.06379720e-01,  3.06379727e-01,  3.06379727e-01,
        3.06379727e-01,  3.06379727e-01])
    
    (result - y_test).sum(axis=1):
    array([ 32.59145649, -72.12072122, -71.80247098, -71.80631274,
       -71.80626636, -71.80626692, -71.80626692, -71.80626692,
       -71.80626692, -71.80626692])
    
    sum(y_test)  : 72.11264664183513
    说明：第一轮迭代后就没什么提升了。
    
    
    """

    lr = LinearRegression(optimize_method='batch_gradient_descent', max_iter=1000, loss='abs_loss', alpha=1,
                                    batch_num=0.2, learning_rate=0.01, min_iter_loss=0.01, random_state=1)
    lr.fit(X, y)
    y_lr_pred = lr.predict(X_test)
    print("lr         Mean squared error: %.6f" % mean_squared_error(y_test, y_lr_pred))

    # # sklearn 自带
    # sklean_gbdt = GradientBoostingRegressor(max_depth=2, min_samples_split=3, n_estimators=10, learning_rate=1)
    # sklean_gbdt.fit(X, y)
    # sklean_pred = sklean_gbdt.predict(X_test)
    # print("sklearn gbr  Mean squared error: %.6f" % mean_squared_error(y_test, sklean_pred))

    plt.scatter(X_test, y_test, color='black')  # 散点输出
    plt.plot(X_test[:, 0], y_lr_pred, color='blue', linewidth=2)  # 预测输出
    plt.plot(X_test[:, 0], y_gblr_pred, color='red', linewidth=2)  # 预测输出
    # plt.scatter(X_test, y_gblr_pred, color='red')  # 预测输出

    # 过程
    for y in gblr.y_list:
        plt.scatter(X_test, y)  # 预测输出

    # plt.plot(X_test[:, 0], y_test-y_gblr_pred, color='red', linewidth=2)  # 预测输出
    plt.show()
