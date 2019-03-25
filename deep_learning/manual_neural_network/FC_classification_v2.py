#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-18 下午8:22
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : FC_classification_v2.py
# @Software: PyCharm
"""
分类、全连接前馈、sigmod激活函数、softmax、交叉熵损失
fully connected
"""
from util.dataset import multi_iris

import numpy as np
from sklearn.metrics import classification_report


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def sigmod_derivative(x):
    return x*(1-x)


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return np.sign(np.maximum(x, 0))


def generate_dataset(ndim=3, batch_size=1000):
    x = np.random.randn(batch_size*ndim).reshape((batch_size, ndim))
    y = np.sum(np.multiply(x, np.arange(ndim)), axis=1) + 0.5*np.random.randn()
    # y = np.sum(np.multiply(x, [2, 3, 4]), axis=1)
    return x, y.reshape((batch_size, 1))


def softmax(x):
    total = np.sum(np.exp(x), axis=1).reshape((-1, 1))
    return np.exp(x) / total


class BackPropagation(object):

    def __init__(self, structure, learning_rate=0.001):
        self.W = []
        self.b = []
        self.structure = structure
        self.layer_num = len(self.structure)
        self.learning_rate = learning_rate
        self.init_network()
        self.batch_size = 10
        self.activation = "relu"
        self.X_test = None
        self.y_test = None

        # 激活函数
        if self.activation == "sigmod":
            self.act_func = sigmod
            self.act_func_derivative = sigmod_derivative
        elif self.activation == "relu":
            # note: 当使用relu时，损失出现越来越大的问题。原来是学习率过高了。之前sigmod设置的0.02。 当改为0.001时，relu正常了。
            self.act_func = relu
            self.act_func_derivative = relu_derivative

    def init_network(self):
        for i in range(self.layer_num-1):
            pre_layer_node_num = self.structure[i]
            curr_layer_node_num = self.structure[i+1]

            self.W.append(np.random.randn(pre_layer_node_num*curr_layer_node_num).reshape((pre_layer_node_num, curr_layer_node_num)))
            self.b.append(np.random.randn(curr_layer_node_num))


    def train(self, X_train, y_train, X_test, y_test):
        self.train_num = X_train.shape[0]
        self.X_test = X_test
        self.y_test = y_test

        for i in range(30000):
            choice_indexes = np.random.choice(self.train_num, replace=True, size=self.batch_size)
            batch_x = X_train[choice_indexes]
            batch_y = y_train[choice_indexes]
            self.train_batch(batch_x, batch_y, iter=i)

    def train_batch(self, batch_x, batch_y, iter):
        # 前向计算output
        forward_outputs = []
        forward_outputs.append(batch_x)
        for i in range(self.layer_num-2):
            forward_outputs.append(self.act_func(forward_outputs[i].dot(self.W[i]) + self.b[i]))

        # 最后一层
        output = softmax(forward_outputs[-1].dot(self.W[-1]) + self.b[-1])

        # 损失
        if iter % 100 == 0:
            training_loss = np.mean(self.calc_loss(output, batch_y, type="cross_entropy"))
            valid_loss = -1
            if self.X_test is not None:
                pred = self.predict(self.X_test)
                valid_loss = np.mean(self.calc_loss(pred, self.y_test, type="cross_entropy"))
            print("iter: %s, training loss: %s, valid loss: %s" % (iter, training_loss, valid_loss))

        # 反向传播
        backward_delta = [[] for _ in range(self.layer_num-1)]
        # 最后一层W更新
        backward_delta[-1] = (output - batch_y)
        self.W[-1] -= self.learning_rate * forward_outputs[-1].T.dot((output - batch_y)) / self.batch_size
        self.b[-1] -= self.learning_rate * np.mean(output - batch_y)

        # 反向传播隐藏层W更新
        for i in range(self.layer_num-2, 0, -1):
            backward = backward_delta[i]
            w = self.W[i].T
            # 当前是多个后向节点与W的乘积之和。 𝜕C/𝜕a;  a = sigmod(z), z = wx+b.
            d = backward.dot(w)
            # 激活函数的导数
            derivative = self.act_func_derivative(forward_outputs[i])
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
        if type == "square":
            return np.square(output-y)
        elif type == "cross_entropy":
            return -np.multiply(y, np.log(output))

    def predict(self, x):
        forward_outputs = []
        forward_outputs.append(x)
        for i in range(self.layer_num-2):
            forward_outputs.append(self.act_func(forward_outputs[i].dot(self.W[i]) + self.b[i]))
        return softmax(forward_outputs[-1].dot(self.W[-1]) + self.b[-1])


if __name__ == "__main__":
    np.random.seed(1)
    ndim = 4
    X_train, X_test, y_train, y_test = multi_iris(top_features=ndim, random_state=3, onehot=True)
    bp = BackPropagation(structure=(ndim, 8, 4, 3))
    bp.train(X_train, y_train, X_test, y_test)

    y_pred = bp.predict(X_test).argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    print(classification_report(y_test, y_pred, target_names=['label:0', 'label:1', 'label:2']))
