#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/3 5:02 PM
# @Author  : yangsen
# @Site    : 
# @File    : GradientBoostRegressionTree.py
# @Software: PyCharm
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from util.dataset import binary_iris


def mean_square(vector):
    """
    TODO:
    在GradientBoostRegressionTree.py中，如果使用 np.sqrt(np.square(vector-vector.mean()).sum()) 则提升不动。
    需要再考虑解决清除该问题！

    :param vector:
    :return:
    """
    return np.square(vector-vector.mean()).sum()


class CARTRegression(object):
    X_train = np.zeros((2, 2))
    y_train = np.zeros(2)
    train_num = 2
    feature_num = 2
    feature_values = []  # 索引为特征序号，值所有可能取值的列表
    tree = {}

    """
    简单版，假设二分类，假设连续变量，只做回归树。
    损失使用 平方损失。
    """
    def __init__(self, leaf_min_samples=3, max_depth=3):
        self.tree = {}

        # 如果节点样本数 小于等于leaf_min_samples 则不再进行分裂
        self.leaf_min_samples = leaf_min_samples
        self.max_depth = max_depth

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # self.labels = Counter(train_label).values()
        self.train_num = self.X_train.shape[0]
        self.feature_num = self.X_train.shape[1]
        self.train_indexes = np.arange(self.train_num)
        self.feature_indexes = np.arange(self.feature_num)

        # todo: 此处暂不做bins优化，使用全局最优查找分裂点
        self.feature_values = [list(set(self.X_train[:, feature_index])) for feature_index in
                               range(self.feature_num)]
        # 建树
        self.tree = self._create_branch(train_indexes=np.arange(self.train_num), depth=0)

    def _leaf_value(self, train_indexes):
        # return self.y_train[train_indexes].mean()
        y = self.y_train[train_indexes]
        value = np.sum(y) / np.sum(np.abs(y)*(1-np.abs(y)))
        return value

    def _create_branch(self, train_indexes, depth):
        depth += 1
        # 不再分裂，返回均值.
        if train_indexes.shape[0] <= self.leaf_min_samples:
            # print("stop split, leaf_min_samples")
            return self._leaf_value(train_indexes)
        if depth > self.max_depth:
            # print("stop split, max_depth")
            return self._leaf_value(train_indexes)

        # 遍历feature，找出增益最大的那个
        loss_gains = np.zeros(self.feature_num)
        loss_split_values = np.zeros(self.feature_num)
        for index, feature_index in enumerate(self.feature_indexes):
            loss_gains[index], loss_split_values[index] = self._max_feature_loss_gain(train_indexes, feature_index)

        # 选中的进行分裂的特征
        choice_feature_index = self.feature_indexes[loss_gains.argmax()]
        max_loss_gain = loss_gains[loss_gains.argmax()]
        best_feature_idx = loss_gains.argmax()
        if max_loss_gain <= 0:  # 如果均无信息增益，直接返回均值
            return self.y_train[train_indexes].mean()

        choice_feature_index = self.feature_indexes[best_feature_idx]
        choice_feature_split_value = loss_split_values[best_feature_idx]

        branch = {}
        branch['feature'] = choice_feature_index
        branch['split_value'] = choice_feature_split_value

        left_branch_train_indexes = train_indexes[self.X_train[train_indexes][:, choice_feature_index] < choice_feature_split_value]
        right_branch_train_indexes = train_indexes[self.X_train[train_indexes][:, choice_feature_index] >= choice_feature_split_value]
        branch['left'] = self._create_branch(left_branch_train_indexes, depth=depth)
        branch['right'] = self._create_branch(right_branch_train_indexes, depth=depth)

        return branch

    def _max_feature_loss_gain(self, train_indexes, feature_index):
        """
        该特征最大的损失增益
        :param train_indexes:
        :param feature_index:
        :return:
        """
        sorted_value = sorted(list(set(self.X_train[train_indexes][:, feature_index])))
        value_num = len(sorted_value)
        # value_num不可能为0，因为train_indexes时需要检查

        # 该特征完全相同，但是lavel完全相同不会进来进行计算。所以，一定不会选这个特征，该特征引入不会增加信息。
        # 如果全部的特征均相同，则不在进行分裂。
        if value_num == 1:
            return 0, 0

        max_loss_gain = 0
        best_split_value = 0
        for end in range(1, value_num):
            start = end - 1
            split_value = (sorted_value[end] + sorted_value[start]) / 2
            # 计算当前分割点的信息增益
            loss_gain = self._loss_gain_split_value(train_indexes=train_indexes, feature_index=feature_index,
                                              split_value=split_value)
            if max_loss_gain < loss_gain:
                max_loss_gain = loss_gain
                best_split_value = split_value
        return max_loss_gain, best_split_value

    def _loss_gain_split_value(self, train_indexes, feature_index, split_value):
        """
        该特征 该分割点的 损失增益
        :param train_indexes:
        :param feature_index:
        :param split_value:
        :return:
        """

        # 混乱，熵大
        origin_loss = mean_square(self.y_train[train_indexes])

        new_loss = 0
        # 计算所有可能取值下，剩余的熵

        y_train_left = self.y_train[train_indexes][self.X_train[train_indexes][:, feature_index] <= split_value]
        y_train_right = self.y_train[train_indexes][self.X_train[train_indexes][:, feature_index] > split_value]
        new_loss = mean_square(y_train_left) + mean_square(y_train_right)
        loss_gain = origin_loss - new_loss
        # if loss_gain < 0:
        #     print(loss_gain)
        return loss_gain

    def predict(self, X_test):
        # 连续变量特征
        def pred_sample(tree, feature_vec):
            if type(tree) != dict:
                return tree
            else:
                if feature_vec[tree['feature']] < tree['split_value']:
                    return pred_sample(tree['left'], feature_vec)
                else:
                    return pred_sample(tree['right'], feature_vec)
        result = np.zeros(X_test.shape[0])
        for index, feature_vec in enumerate(X_test):
            result[index] = pred_sample(self.tree, feature_vec)
        return result


class GBCT(object):

    def __init__(self, learning_rate=1.0, subsample=1, n_estimators=10, min_samples_split=3, max_depth=2):
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.cart_trees = []

    def _calc_gradient(self, sum_y, y):
        return y / (1 + np.exp(y*sum_y))

    def fit(self, X_train, y_train):
        X = X_train
        # y = y_train
        # sum_y = np.zeros(y_train.shape[0])
        sum_y = np.ones(y_train.shape[0])/2
        for i in range(self.n_estimators):
            gradient = self._calc_gradient(y_train, sum_y)
            cart = CARTRegression(max_depth=self.max_depth, leaf_min_samples=self.min_samples_split)
            cart.fit(X, gradient)
            self.cart_trees.append(cart)
            y_pred = cart.predict(X_train)
            sum_y += y_pred * self.learning_rate
            print("Tree %s        accuracy_score: %s" % (i, accuracy_score(np.sign(sum_y).astype('int'), y_train)))
            # print("Tree %s        Mean squared error: %.6f" % (i, mean_squared_error(sum_y, y_train)))

    def predict(self, X_test):
        y = np.zeros(X_test.shape[0])
        for cart in self.cart_trees:
            y += cart.predict(X_test) * self.learning_rate
        return y


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = binary_iris(type=1, top_features=4, random_state=6)

    gbct = GBCT(max_depth=3, min_samples_split=2, n_estimators=10, learning_rate=.1)
    gbct.fit(X_train, y_train)
    y_gbct_pred = gbct.predict(X_test)
    print("gbct        accuracy_score: %s" % (accuracy_score(np.sign(y_gbct_pred).astype('int'), y_test)))

    # sklearn 自带
    sklean_gbdt = GradientBoostingClassifier(max_depth=3, min_samples_split=2, n_estimators=10, learning_rate=0.1)
    sklean_gbdt.fit(X_train, y_train)
    sklean_pred = sklean_gbdt.predict(X_test)
    print("sklearn gbr  accuracy_score: %.6f" % accuracy_score(y_test, sklean_pred))
