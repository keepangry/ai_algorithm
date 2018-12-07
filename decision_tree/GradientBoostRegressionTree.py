# encoding: utf-8
'''
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: yangsen
@software: garner
@file: CARTRegression.py
@time: 18-8-23 下午9:12
@desc:
CART:  Classification And Regression Tree  分类回归树


如果是分类树，CART采用GINI值衡量节点纯度；如果是回归树，采用样本方差衡量节点纯度。
CART都要选择使子节点的GINI值或者回归方差最小的属性作为分裂的方案

但CART是一棵二叉树，每一次分裂只会产生两个节点，怎么办呢？很简单，只要将其中一个离散值独立作为一个节点，其他的离散值生成另外一个节点即可。

CART跟C4.5一样，需要进行剪枝，采用CCP（代价复杂度的剪枝方法）。

## 模型树： 采用线性回归的最小均方损失来计算该节点的损失。  叶子节点的值可以是均值、中值、众数等。最终计算的是均方误差最小。


# 剪枝，有一个参数a， 对树的复杂度 与 损失的均衡。  既需要损失小，又希望树不过于复杂导致过拟合。设置超参a。
# 建立完成后进行计算，如果合并满足条件，则进行剪枝。
# 或者在生成时，就设置树的深度。 或 增益率过低时提前终止。 或者 叶子节点样本最小数量。


理解：模型树和回归树的区别就是回归树的叶节点是一个常数值，而模型树的叶节点是分段线性函数，
分段线性模型就是我们对数据集的一部分数据以某个线性模型建模，而另一份数据以另一个线性模型建模。

模型树与回归树的差别在于：回归树的叶节点是节点数据标签值的平均值，而模型树的节点数据是一个线性模型（可用最简单的最小二乘法来构建线性模型），
返回线性模型的系数W，我们只要将测试数据X乘以W便可以得到预测值Y，即Y=X*W。所以该模型是由多个线性片段组成的。


'''

from collections import Counter
import numpy as np
# from util.loss import mean_square


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


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

    def _create_branch(self, train_indexes, depth):
        depth += 1
        # 不再分裂，返回均值.
        if train_indexes.shape[0] <= self.leaf_min_samples:
            # print("stop split, leaf_min_samples")
            return self.y_train[train_indexes].mean()
        if depth > self.max_depth:
            # print("stop split, max_depth")
            return self.y_train[train_indexes].mean()

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


if __name__ == "__main__":
    """
        
    """

    # diabetes = datasets.load_diabetes()
    #
    # # Use only one feature
    # diabetes_X = diabetes.data[:, np.newaxis, 2]
    #
    # # Split the data into training/testing sets
    # diabetes_X_train = diabetes_X[:-20]
    # diabetes_X_test = diabetes_X[-20:]
    #
    # # Split the targets into training/testing sets
    # diabetes_y_train = diabetes.target[:-20]
    # diabetes_y_test = diabetes.target[-20:]
    #
    # regr = linear_model.LinearRegression()
    # regr.fit(diabetes_X_train, diabetes_y_train)
    # diabetes_y_pred = regr.predict(diabetes_X_test)
    # print("Mean squared error: %.2f"
    #       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    #
    #
    # ##
    # cart = CARTRegression()
    # cart.fit(diabetes_X_train, diabetes_y_train)
    # diabetes_y_pred = cart.predict(diabetes_X_test)
    # print("Mean squared error: %.2f"
    #       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    #
    # plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    # plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()



    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    # y[::5] += 3 * (0.5 - rng.rand(16))

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_test = np.sin(X_test).ravel()
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    cart = CARTRegression()
    cart.fit(X, y)
    y_cart_pred = cart.predict(X_test)


    # Plot the results
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black",
                c="darkorange", label="data")
    plt.plot(X_test, y_1, color="red", label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_cart_pred, color="cornflowerblue", label="cart", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

    #
    print("cart        Mean squared error: %.6f" % mean_squared_error(y_test, y_cart_pred))
    print("max_depth=2 Mean squared error: %.6f" % mean_squared_error(y_test, y_1))
    print("max_depth=5 Mean squared error: %.6f" % mean_squared_error(y_test, y_2))
