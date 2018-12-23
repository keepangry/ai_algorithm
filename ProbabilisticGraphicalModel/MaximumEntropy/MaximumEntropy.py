# encoding: utf-8
'''
@author: yangsen
@license:
@contact: 0@keepangry.com
@software:
@file: MaximumEntropy.py
@time: 18-10-2 上午7:56
@desc:
'''
from sklearn.datasets import load_digits
import numpy as np
import math

class MaximumEntropy(object):

    def __init__(self):
        self.feature_name_prefix_num = 1000000
        self.feature_name_delimiter = ":"

        self.max_iter = 1000
        self.eps = 0.01   # 判断单维度收敛阈值


    def feature_transfrom(self, features, feature_names):
        return np.char.add(np.char.add(feature_names, self.feature_name_delimiter), features)



    def fit(self, X_train, y_train, feature_names=None):
        """

        :param X_train: feature_names=None时，为二维变长列表。 feature_names为特征列表时，为二维定长数组。
        :param y_train:
        :param feature_names:
        :return:
        """
        # 1、 特征转换。 最大熵算的是(x,y)，如果只使用特征值，会导致特征值错乱，需要把特征值改为 feature_name+feature_value
        if feature_names is not None:
            self.X_train = self.feature_transfrom(np.array(X_train).astype('str'), np.array(feature_names).astype('str'))
        else:
            self.X_train = X_train
            # feature_names = (self.feature_name_prefix_num + np.arange(X_train.shape[1])).astype('str')
        self.y_train = y_train

        self.train_num = len(self.X_train)
        # self.feature_num = self.X_train.shape[1]

        # 在定长情况下
        # self.M = self.feature_num

        # 计算所有y
        self.Y = list(set(self.y_train))

        # 所有的(x,y) 存入字典
        self.xy2idx = {}
        index = 0
        self.xy_num_array = []

        self.x2idx = {}
        x_index = 0
        self.x_num_array = []
        for instance_idx in range(self.train_num):
            for feature_value in self.X_train[instance_idx]:
                # xy
                xy = (feature_value, self.y_train[instance_idx])
                if xy not in self.xy2idx:
                    self.xy2idx[xy] = index
                    index += 1
                    self.xy_num_array.append(1)
                else:
                    self.xy_num_array[self.xy2idx[xy]] += 1

                # x
                x = feature_value
                if x not in self.x2idx:
                    self.x2idx[x] = x_index
                    x_index += 1
                    self.x_num_array.append(1)
                else:
                    self.x_num_array[self.x2idx[x]] += 1
        self.x_num_array = np.array(self.x_num_array)
        self.x_num = self.x_num_array.shape[0]

        # (x,y)出现的次数数组
        self.xy_num_array = np.array(self.xy_num_array)

        # 不同的(x,y)数量
        self.xy_num = self.xy_num_array.shape[0]

        # 总的(x,y)数量
        self.N = sum(self.xy_num_array)

        # 常数M
        self.M = self.xy_num

        # 初始化W
        self.W = np.zeros(self.xy_num)

        # 计算 (x,y)经验期望
        self.P_xy_ = self.xy_num_array / (self.train_num*1.0)


        # 迭代
        for iter_num in range(self.max_iter):
            print('iter_num:',iter_num)
            # 是否全部w都收敛 标志
            stop_flag = True

            ##### 计算 (x,y)模型期望
            self._model_ep()

            # 对每一个w进行更新，判断是否收敛
            for w_idx in range(self.xy_num):
                δ = 1.0 / self.xy_num * math.log(self.P_xy_[w_idx] / self.P_xy[w_idx])
                # 计算w的更新值
                if abs(δ) > self.eps:
                    self.W[w_idx] += δ
                    stop_flag = False
            # print(self.W)
            if stop_flag:
                print("iter stop.")
                break

    def P_yx(self, X):
        zx = 0.0
        for y in self.Y:
            y_w_sum = 0.0
            for feature_value in X:
                xy = (feature_value, y)
                if xy in self.xy2idx:
                    y_w_sum += self.W[self.xy2idx[xy]]
            print(y_w_sum)
            zx += math.exp(y_w_sum)

        P_yx = {}
        for y in self.Y:
            y_w_sum = 0.0
            for feature_value in X:
                xy = (feature_value, y)
                if xy in self.xy2idx:
                    y_w_sum += self.W[self.xy2idx[xy]]
            p_yx = 1.0 / zx * math.exp(y_w_sum)
            P_yx[y] = p_yx
        return P_yx


    def _model_ep(self):
        self.P_xy = np.zeros(self.xy_num)

        zx_array = np.zeros(self.train_num)
        for instance_idx in range(self.train_num):
            P_yx = self.P_yx(self.X_train[instance_idx])

            for y, p_yx in P_yx.items():
                ## (y, p_yx) 计算模型ep
                for feature_value in self.X_train[instance_idx]:
                    xy = (feature_value, y)
                    if xy in self.xy2idx:
                        self.P_xy[self.xy2idx[xy]] += 1.0 / self.train_num * p_yx

    def predict(self, X):
        return self.P_yx(X)


if __name__ == "__main__":
    # [天气、温度、时间、是否有伴、是否有车]
    feature_names = np.array(['weather','temperature','time','is_together','is_car'])
    features = np.array([
        # ['sunny','hot','morning', 1, 0],
        # ['sunny','cool','noon', 0, 0],
        # ['overcast','cold','night', 1, 0],
        # ['overcast','hot','morning', 0, 1],
        # ['rainy','cool','noon', 0, 1],
        # ['rainy','cold','night', 1, 1],
        ['sunny', 'hot', 'morning'],
        ['sunny', 'cool', 'noon'],
        ['overcast', 'cold', 'night'],
        ['overcast', 'hot', 'morning'],
        ['rainy', 'cool', 'noon'],
        ['rainy', 'cold', 'night'],
    ])
    # TODO： 最大熵可以是变长的。比如，文章中的词。

    # 是否出门
    labels = np.array([
        1,1,0,1,0,0
    ])

    maxent = MaximumEntropy()
    maxent.fit(features, labels)
    # maxent.fit(features, labels, feature_names=feature_names)
    print(maxent.predict(['sunny','hot','noon','sunny']))
    print(maxent.predict(['sunny','hot','noon']))
    print(maxent.predict(['sunny','cold','noon']))
    print(maxent.predict(['rainy','cold','night']))




