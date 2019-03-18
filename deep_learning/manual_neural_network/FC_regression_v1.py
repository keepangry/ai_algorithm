#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/18 9:53 AM
# @Author  : yangsen
# @Site    : 
# @File    : FC_regression_v1.py
# @Software: PyCharm
"""
回归、全连接前馈、平方损失
fully connected
"""
import numpy as np
from sklearn.model_selection import train_test_split

def sigmod(x):
    return 1 / (1 + np.exp(-x))


def sigmod_derivative(x):
    return x*(1-x)


def generate_dataset(ndim=3, batch_size=1000):
    x = np.random.randn(batch_size*ndim).reshape((batch_size, ndim))
    y = np.sum(np.multiply(x, np.arange(ndim)), axis=1) + 0.5*np.random.randn()
    # y = np.sum(np.multiply(x, [2, 3, 4]), axis=1)
    return x, y.reshape((batch_size, 1))


class back_propagation(object):

    W = []
    b = []

    def __init__(self, structure):
        self.structure = structure
        self.layer_num = len(self.structure)
        self.learning_rate = 0.02
        self.init_network()
        self.batch_size = 10

    def init_network(self):
        for i in range(self.layer_num-1):
            pre_layer_node_num = self.structure[i]
            curr_layer_node_num = self.structure[i+1]

            self.W.append(np.random.randn(pre_layer_node_num*curr_layer_node_num).reshape((pre_layer_node_num, curr_layer_node_num)))
            self.b.append(np.random.randn(curr_layer_node_num))

    def train(self, X_train, y_train, X_test, y_test):
        self.train_num = X_train.shape[0]

        for i in range(30000):
            choice_indexes = np.random.choice(self.train_num, replace=False, size=self.batch_size)
            batch_x = X_train[choice_indexes]
            batch_y = y_train[choice_indexes]
            self.train_batch(batch_x, batch_y, iter=i)

    def train_batch(self, batch_x, batch_y, iter):
        # 前向计算output
        forward_outputs = []
        forward_outputs.append(batch_x)
        for i in range(self.layer_num-2):
            forward_outputs.append(sigmod(forward_outputs[i].dot(self.W[i]) + self.b[i]))

        # 最后一层
        output = forward_outputs[-1].dot(self.W[-1]) + self.b[-1]

        # 损失
        if iter % 100 == 0:
            training_loss = np.mean(self.calc_loss(output, batch_y))
            pred = self.predict(X_test)
            valid_loss = np.mean(self.calc_loss(pred, y_test))
            print("iter: %s, training loss: %s, valid loss: %s" % (iter, training_loss, valid_loss))

        # 反向传播
        backward_delta = [[] for _ in range(self.layer_num-1)]
        # 最后一层W更新
        backward_delta[-1] = (output - batch_y) * 1  # 𝜕C/𝜕y。 此处无激活函数，所以为1
        self.W[-1] -= self.learning_rate * np.mean(np.multiply((output - batch_y), forward_outputs[-1]))
        self.b[-1] -= self.learning_rate * np.mean(output - batch_y)

        # 反向传播隐藏层W更新
        for i in range(self.layer_num-2, 0, -1):
            backward = backward_delta[i]
            w = self.W[i].T
            # 当前是多个后向节点与W的乘积之和。 𝜕C/𝜕a;  a = sigmod(z), z = wx+b.
            d = backward.dot(w)
            # 激活函数的导数
            derivative = sigmod_derivative(forward_outputs[i])
            backward_delta[i-1] = np.multiply(d, derivative)

            # 对当前W的梯度为：前向 与 后向 相乘。
            gradients = np.zeros((self.batch_size, self.structure[i-1], self.structure[i]))
            for sample_idx in range(self.batch_size):
                backward = backward_delta[i-1][sample_idx].reshape(-1, 1)
                forward = forward_outputs[i-1][sample_idx].reshape(1, -1)
                gradients[sample_idx] = backward.dot(forward).T
            gradient = np.mean(gradients, axis=0)
            self.W[i-1] -= self.learning_rate * gradient
            self.b[i-1] -= self.learning_rate * np.sum(backward_delta[i-1], axis=0)

    def calc_loss(self, output, y, type="square"):
        return np.square(output-y)

    def predict(self, x):
        forward_outputs = []
        forward_outputs.append(x)
        for i in range(self.layer_num-2):
            forward_outputs.append(sigmod(forward_outputs[i].dot(self.W[i]) + self.b[i]))
        return forward_outputs[-1].dot(self.W[-1]) + self.b[-1]


if __name__ == "__main__":
    np.random.seed(1)
    ndim = 6
    bp = back_propagation(structure=(ndim, 16, 8, 4, 1))
    X, y = generate_dataset(ndim=ndim)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    bp.train(X_train, y_train, X_test, y_test)
