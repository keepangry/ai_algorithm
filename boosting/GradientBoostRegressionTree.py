#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/3 5:02 PM
# @Author  : yangsen
# @Site    : 
# @File    : GradientBoostRegressionTree.py
# @Software: PyCharm
import numpy as np
from decision_tree.CARTRegression import CARTRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor


class GBRT(object):

    def __init__(self, learning_rate=1.0, subsample=1, n_estimators=10, min_samples_split=3, max_depth=2):
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.cart_trees = []

    # def fit(self, X_train, y_train):
    #     X = X_train
    #     y = y_train
    #     for i in range(self.n_estimators):
    #         cart = CARTRegression(max_depth=self.max_depth, leaf_min_samples=self.min_samples_split)
    #         cart.fit(X, y)
    #         self.cart_trees.append(cart)
    #         y_pred = cart.predict(X_train)
    #         print("Tree %s        Mean squared error: %.6f" % (i, mean_squared_error(y, y_pred)))
    #         # 当为平方损失时，即拟合残差(y-pred) /  负梯度-(pred-y)
    #         y = -(y_pred - y)*self.learning_rate

    def fit(self, X_train, y_train):
        X = X_train
        y = y_train
        sum_y = np.zeros(y_train.shape[0])
        for i in range(self.n_estimators):
            cart = CARTRegression(max_depth=self.max_depth, leaf_min_samples=self.min_samples_split)
            cart.fit(X, y)
            self.cart_trees.append(cart)
            y_pred = cart.predict(X_train)
            # 当为平方损失时，即拟合残差(y-pred) /  负梯度-(pred-y)
            # y = -(y_pred - y)*self.learning_rate
            sum_y += y_pred
            y = -(sum_y - y_train) * self.learning_rate
            print("Tree %s        Mean squared error: %.6f" % (i, mean_squared_error(sum_y, y_train)))

    def predict(self, X_test):
        y = np.zeros(X_test.shape[0])
        for cart in self.cart_trees:
            y += cart.predict(X_test)
        return y


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()

    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_test = np.sin(X_test).ravel()

    gbrt = GBRT(max_depth=2, min_samples_split=3, n_estimators=10, learning_rate=0.8)
    gbrt.fit(X, y)
    y_gbrd_pred = gbrt.predict(X_test)
    print("gbrt         Mean squared error: %.6f" % (mean_squared_error(y_test, y_gbrd_pred)))

    cart = CARTRegression(max_depth=2, leaf_min_samples=3)
    cart.fit(X, y)
    y_cart_pred = cart.predict(X_test)
    print("cart         Mean squared error: %.6f" % mean_squared_error(y_test, y_cart_pred))

    # sklearn 自带
    sklean_gbdt = GradientBoostingRegressor(max_depth=2, min_samples_split=3, n_estimators=10, learning_rate=0.8)
    sklean_gbdt.fit(X, y)
    sklean_pred = sklean_gbdt.predict(X_test)
    print("sklearn gbr  Mean squared error: %.6f" % mean_squared_error(y_test, sklean_pred))
