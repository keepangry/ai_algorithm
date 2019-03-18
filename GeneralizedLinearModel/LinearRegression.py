# encoding: utf-8
'''
@author: yangsen
@license:
@contact: 0@keepangry.com
@software:
@file: GeneralizedLinearModel.py
@time: 18-9-1 下午10:41
@desc:

LASSO: The Least Absolute Shrinkage and Selection Operator
'''

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model



class LinearRegression(object):
    def __init__(self, method="gradient_descent", max_iter=10000, loss='square_loss', learning_rate=0.001, min_iter_loss=0.01,
                 optimize_method='gradient_descent', batch_num=0.01, random_state=0, alpha=1.0):
        """

        :param method:
        :param max_iter:
        :param loss: (square_loss, abs_loss, ridge, lasso) ridge: square_loss + L2 , lasso square_loss + L1
        :param learning_rate:
        :param min_iter_loss:
        :param optimize_method: (gradient_descent, stochastic_gradient_descent, batch_gradient_descent)
        :param batch_num: (0,1) 代表比例， 大于1 代表样本数量
        :param random_state:
        :param alpha: when loss in (ridge, lasso) alpha代表正则项系数
        """

        self.w = None
        self.method = method

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


    def fit(self, X, y):
        self.train_num = X.shape[0]
        self.feature_num = X.shape[1]

        if self.method == 'least_square':
            self.fit_by_least_square(X, y)
        elif self.method == 'gradient_descent':
            self.fit_by_gradient_descent(X, y)

    def gradient_descent_sample_choice(self, X, y):
        if self.optimize_method == 'gradient_descent':
            curr_X = X
            curr_y = y
        elif self.optimize_method == 'stochastic_gradient_descent':
            choice_index = np.random.randint(self.train_num)
            curr_X = X[[choice_index]]
            curr_y = y[choice_index]
        elif self.optimize_method == 'batch_gradient_descent':
            if 0 < self.batch_num < 1:
                # 无放回
                choice_indexes = np.random.choice(self.train_num, replace=False, size=int(self.batch_num*self.train_num))
            elif 1 <= self.batch_num < self.train_num:
                choice_indexes = np.random.choice(self.train_num, replace=False, size=self.batch_num)
            else:
                raise Exception("batch_num 参数错误")
            curr_X = X[choice_indexes]
            curr_y = y[choice_indexes]

        return curr_X, curr_y

    def gradient(self, theta, curr_hx, curr_X, curr_y):
        if self.loss == 'square_loss':
            gradient = (curr_hx - curr_y).dot(curr_X)
        elif self.loss == 'abs_loss':
            gradient = np.array( list(map(lambda x:1 if x>0 else -1,(curr_hx - curr_y))) ).dot(curr_X)
        elif self.loss == 'ridge':
            gradient = (curr_hx - curr_y).dot(curr_X) + self.alpha * theta
        elif self.loss == 'lasso':
            # alpha需要正负号
            gradient = (curr_hx - curr_y).dot(curr_X) + self.alpha * np.sign(theta)
        return gradient

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
        np.random.seed(self.random_state) # 注意此处只有一次有效
        theta = np.random.random(X.shape[1])
        pre_loss = 9999999999

        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1

            # 选择当前进行优化的样本
            # 全样本梯度下降、随机梯度下降、批梯度下降
            curr_X, curr_y = self.gradient_descent_sample_choice(X, y)

            # 参数为当前theta时的预测值
            curr_hx = curr_X.dot(theta)

            # 求梯度
            # TODO: 抽象出来，根据损失函数进行计算梯度
            gradient = self.gradient(theta, curr_hx, curr_X, curr_y)

            # 更新参数
            theta -= self.learning_rate * gradient

            # 判断是否收敛，方法1：参数的变化是否以及过小
            # 方法2： 损失是否已经不在降低
            # 计算平方损失，即二阶范数的平方
            curr_loss = sum((X.dot(theta) - y)**2)
            # print("iter_num: {}, square loss: {}".format(iter_num, curr_loss))

            # 随机梯度下降经常会导致损失增加，因为他能跳出局部最小值点，该提前终止方法在随机梯度下降时略有问题
            # if pre_loss - curr_loss < self.min_iter_loss:
            #     break
            # else:
            #     pre_loss = curr_loss
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
    X = diabetes.data[:, [8]]
    y = diabetes.target

    # 人造X
    # X = np.arange(0., 10., 0.2).reshape(50,1)
    # y = ( 2 * X + 5 + np.random.randn(1) ).reshape(50,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # 最小二乘
    # lr = GeneralizedLinearModel('least_square')

    # 全样本梯度下降
    # lr = GeneralizedLinearModel()

    # 随机梯度下降
    # lr = GeneralizedLinearModel(optimize_method='stochastic_gradient_descent', learning_rate=0.01, min_iter_loss=0.000001)
    # lr = GeneralizedLinearModel(optimize_method='batch_gradient_descent', batch_num=0.1, learning_rate=0.01, min_iter_loss=0.000001)
    # lr = GeneralizedLinearModel(optimize_method='batch_gradient_descent', loss='square_loss', alpha=1, batch_num=0.2, learning_rate=0.01, min_iter_loss=1)
    lr = LinearRegression(optimize_method='batch_gradient_descent', loss='abs_loss', alpha=1, batch_num=0.2, learning_rate=0.1, min_iter_loss=1)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)


    # cls = linear_model.Lasso(alpha=1.0)
    cls = linear_model.LinearRegression()
    cls.fit(X_train, y_train)
    y_pred_sys = cls.predict(X_test)


    plt.scatter(X_test, y_test, color='black')  # 散点输出
    plt.plot(X_test[:, 0], y_pred, color='blue', linewidth=2)  # 预测输出
    plt.plot(X_test[:, 0], y_pred_sys, color='red', linewidth=2)  # 预测输出
    plt.show()
