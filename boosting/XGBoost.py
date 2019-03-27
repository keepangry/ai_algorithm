#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/7 10:50 AM
# @Author  : yangsen
# @Site    : 
# @File    : XGBoost.py
# @Software: PyCharm

"""
参考：https://github.com/lancifollia/tinygbt
https://www.jianshu.com/p/7467e616f227
https://xgboost.readthedocs.io/en/latest/tutorials/model.html

1、每棵树提升什么
2、分裂与建树的选取
3、约束

"""
import numpy as np
from numpy import square, sum
import matplotlib.pyplot as plt
import xgboost as xgb
from util.loss import root_mean_square


class RegressionTree(object):
    """
    简单版，假设二分类，假设连续变量，只做回归树。
    损失使用 平方损失。
    """
    def __init__(self, leaf_min_samples=3, max_depth=2):
        self.tree = {}
        # 如果节点样本数 小于等于leaf_min_samples 则不再进行分裂
        self.leaf_min_samples = leaf_min_samples
        self.max_depth = max_depth

    def fit(self, X_train, y_train, grad, hessian, params):
        self.X_train = X_train
        self.y_train = y_train
        self.grad = grad
        self.hessian = hessian
        self.params = params


        self.train_num = self.X_train.shape[0]
        self.feature_num = self.X_train.shape[1]
        self.train_indexes = np.arange(self.train_num)
        self.feature_indexes = np.arange(self.feature_num)

        # todo: 此处暂不做bins优化，使用全局最优查找分裂点
        self.feature_values = [list(set(self.X_train[:, feature_index])) for feature_index in
                               range(self.feature_num)]
        # 建树
        self.tree = self._create_branch(train_indexes=np.arange(self.train_num), depth=0)

    def _compute_leaf_value(self, sample_indexes):
        """
        计算叶子值。value即weight。
        :return:
        """
        return sum(self.grad[sample_indexes]) / sum(self.hessian[sample_indexes] + self.params['λ'])

    def _create_branch(self, train_indexes, depth):
        depth += 1
        # 不再分裂，返回均值.
        if train_indexes.shape[0] <= self.leaf_min_samples:
            # print("stop split, leaf_min_samples")
            return self._compute_leaf_value(train_indexes)
        if depth > self.max_depth:
            # print("stop split, max_depth")
            return self._compute_leaf_value(train_indexes)

        # 遍历feature，找出增益最大的那个
        loss_gains = np.zeros(self.feature_num)
        loss_split_values = np.zeros(self.feature_num)
        for index, feature_index in enumerate(self.feature_indexes):
            loss_gains[index], loss_split_values[index] = self._max_feature_loss_gain(train_indexes, feature_index)

        # 选中的进行分裂的特征
        max_loss_gain = loss_gains[loss_gains.argmax()]
        best_feature_idx = loss_gains.argmax()
        if max_loss_gain <= 0:  # 如果均无信息增益，直接返回均值
            return self._compute_leaf_value(train_indexes)

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

        # 未达到最小损失增益，不再进行分裂
        if max_loss_gain < self.params['min_split_loss']:
            return 0, 0

        return max_loss_gain, best_split_value

    def _loss_gain_split_value(self, train_indexes, feature_index, split_value):
        """
        计算分裂增益，obj衡量结构损失
        Gain = obj_left + obj_right - obj_total

        :param train_indexes:
        :param feature_index:
        :param split_value:
        :return:
        """
        grad_total = self.grad[train_indexes]
        grad_left = self.grad[train_indexes][self.X_train[train_indexes][:, feature_index] <= split_value]
        grad_right = self.grad[train_indexes][self.X_train[train_indexes][:, feature_index] > split_value]

        hessian_total = self.hessian[train_indexes]
        hessian_left = self.hessian[train_indexes][self.X_train[train_indexes][:, feature_index] <= split_value]
        hessian_right = self.hessian[train_indexes][self.X_train[train_indexes][:, feature_index] > split_value]

        def calc_obj(g, h):
            return square(sum(g)) / (sum(h) + self.params['λ'])

        # 公式前面还要除以2，因为是计算最大，可以不用进行计算，不影响结果
        gain = calc_obj(grad_left, hessian_left) + calc_obj(grad_right, hessian_right) - \
            calc_obj(grad_total, hessian_total) - self.params['γ']
        return gain

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


class XGBoost(object):

    def __init__(self, max_depth=2,
                 n_estimators=10, learning_rate=0.1, λ=1, γ=0, min_split_loss=0.0, verbose=True):
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.params = {
            'λ': λ,
            'γ': γ,
            'min_split_loss': min_split_loss
        }
        self.trees = []

    def _compute_gradient(self, y_train, curr_y_pred):
        """
        计算梯度和二阶导

        :return:
        """
        # 平方损失的二阶导为2
        hessian = np.full(y_train.shape[0], 2)
        # 平方损失的一阶导
        grad = - 2 * (curr_y_pred - y_train)
        return grad, hessian

    def fit(self, X_train, y_train, eval_set=None):
        self.X_train = X_train
        self.y_train = y_train
        self.sample_num = self.X_train.shape[0]

        if eval_set is not None:
            self.X_eval = eval_set[0]
            self.y_eval = eval_set[1]
            sum_eval_y_pred = np.zeros(self.X_eval.shape[0])

        sum_y_pred = np.zeros(self.sample_num)
        boost_y = self.y_train

        for i in range(self.n_estimators):
            grad, hessian = self._compute_gradient(self.y_train, curr_y_pred=sum_y_pred)
            regression_tree = RegressionTree(max_depth=self.max_depth)
            regression_tree.fit(self.X_train, boost_y, grad, hessian, self.params)

            # TODO: 提前终止条件判断
            self.trees.append(regression_tree)
            sum_y_pred += regression_tree.predict(self.X_train)
            sum_eval_y_pred = 0
            if eval_set is not None:
                sum_eval_y_pred += regression_tree.predict(self.X_eval)

            if self.verbose:
                print("round: %s, current tree rmse: %s, eval rmse: %s" % (i, root_mean_square(self.y_train, sum_y_pred),
                                                                         root_mean_square(self.y_eval, sum_eval_y_pred)))

            boost_y = self.learning_rate * (self.y_train - sum_y_pred)

    def predict(self, X_test):
        y = np.zeros(X_test.shape[0])
        for tree in self.trees:
            y += tree.predict(X_test)
        return y


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()

    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_test = np.sin(X_test).ravel()

    my_xgb = XGBoost(learning_rate=0.1)
    my_xgb.fit(X, y, eval_set=(X_test, y_test))
    my_y_pred = my_xgb.predict(X_test)

    clf = xgb.XGBRegressor(n_estimators=20, learning_rate=0.1, tree_method='exact',
                           objective="reg:linear", eval_metric='rmse', verbose=False)

    clf.fit(X, y, eval_set=[(X_test, y_test)])
    # clf.fit(X, y)
    y_pred = clf.predict(X_test)

    print("My XGBoost  root Mean squared error: %.6f" % root_mean_square(y_test, my_y_pred))
    print("XGBoost     root Mean squared error: %.6f" % root_mean_square(y_test, y_pred))

    plt.figure()
    plt.scatter(X_test, y_test, s=20, edgecolor="black",
                c="darkorange", label="data")
    plt.plot(X_test, my_y_pred, color="red", label="My XGBoost", linewidth=2)
    plt.plot(X_test, y_pred, color="cornflowerblue", label="XGBoost", linewidth=2)

    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

