#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  : yangsen
# @Site    : 
# @File    : AdaBoosting.py
# @Software: PyCharm
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import logging
# logging.basicConfig(level=logging.WARNING)


class AdaBoosting(object):
    def __init__(self, verbose=0, n_estimators=10):
        self.n_estimators = n_estimators
        if verbose == 0:
            logging.basicConfig(level=logging.NOTSET)

    def fit(self, instances, labels):
        assert instances.shape[0] != labels[0]
        self.instances = instances
        self.labels = labels
        self.instances_num = instances.shape[0]
        self.instances_dim = instances.shape[1]
        weights = np.full(self.instances_num, 1/self.instances_num)

        # 减少决策桩计算。对样本各维度数值进行排序，并配置对应的label
        # 找到label反转的位置。该位置才需要计算决策结果
        self._decision_preprocess()

        # 所有弱分类器
        self.decision_infos = []
        # 所有弱分类器的权重
        self.decision_weights = []
        # 所有步骤的样本权重，可以不存。
        self.step_weights = [weights]
        for i in range(self.n_estimators):
            decision_info = self.decision_stump(weights)
            logging.info(decision_info)
            self.decision_infos.append(decision_info)
            logging.info("min_error_rate: %s" % decision_info['min_error_rate'])
            decision_weight = 0.5 * np.log((1-decision_info['min_error_rate']) / (decision_info['min_error_rate']+1e-16))
            logging.info("decision_weight: %s" % decision_weight)

            self.decision_weights.append(decision_weight)
            # 对样本集进行预测，计算错误率，以及错误的样本，调整样本权重
            decision_label = self._decision_stump_instance_result(decision_info, instances=self.instances)
            decision_result = decision_label == self.labels

            # 分类正确的平分0.5，分类错误的评分0.5
            true_num = np.sum(decision_result)
            logging.info("true_num: %s" % true_num)

            # 标准weight计算
            new_weights = weights * np.exp(- decision_weight * decision_label * self.labels)
            weights = new_weights / new_weights.sum()

            # 升级版
            # 错误的平分0.5权重，正确的按原权重比例分0.5权重。
            # new_weights = np.zeros(self.instances_num)
            # if self.instances_num-true_num != 0:
            #     new_weights[decision_result == False] = 0.5/(self.instances_num-true_num)
            # new_weights[decision_result] = 0.5 * weights[decision_result] / weights[decision_result].sum()
            # weights = new_weights

            logging.info("weights: %s" % weights)
            logging.info("weights_sum: %s" % weights.sum())

            error_rate = self.error_rate(instances=self.instances, labels=self.labels)
            logging.info("error rate: %s", error_rate)
            if error_rate == 0:
                break
            logging.info("--"*20)
            self.step_weights.append(weights)

        # 各决策树权重
        logging.info(self.decision_weights)
        logging.info("training finished!")

    def error_rate(self, instances, labels):
        error_rate = 1 - np.sum(self.predict(instances) == labels) / instances.shape[0]
        return error_rate

    def predict(self, instances):
        predict = np.zeros(instances.shape[0])
        for i in range(len(self.decision_infos)):
            decision_info = self.decision_infos[i]
            decision_weight = self.decision_weights[i]
            predict += decision_weight*self._decision_stump_instance_result(decision_info, instances)
        pred_result = np.sign(predict)
        # 如果正好为0，则设置默认为1
        pred_result[pred_result == 0] = 1
        return pred_result

    def _decision_stump_instance_result(self, decision_info, instances):
        """

        :param decision_info:
        :return:  [true, false, true]  #分类正确与错误
        """
        feature = instances.T[decision_info['decision_feature_index']]
        decision_feature_value = decision_info['decision_feature_value']

        decision_label = np.zeros(instances.shape[0])
        if decision_info['decision_type'] == '<':
            decision_label[feature < decision_feature_value] = 1
            decision_label[feature >= decision_feature_value] = -1
        else:
            decision_label[feature >= decision_feature_value] = 1
            decision_label[feature < decision_feature_value] = -1
        return decision_label

    def _decision_preprocess(self):
        """
        决策依据预计算
        :return:
        """
        feature_matrix = np.zeros((self.instances_dim, self.instances_num))
        label_matrix = np.zeros((self.instances_dim, self.instances_num))
        argsort_matrix = np.zeros((self.instances_dim, self.instances_num))

        features_decision_pos = []
        for i in range(self.instances_dim):
            dim = self.instances[:, i]
            tmp = np.stack((dim, self.labels))
            argsort = np.argsort(dim)
            dim_sort = tmp.T[argsort].T
            # dim_sort = np.array(sorted(tmp.T, key=(lambda x: x[0]))).T

            argsort_matrix[i] = argsort
            feature_matrix[i] = dim_sort[0]
            label_matrix[i] = dim_sort[1]
            # 找到label反转的位置
            feature_decision_pos = []
            for idx in range(1, self.instances_num):
                if label_matrix[i][idx-1] != label_matrix[i][idx]:
                    feature_decision_pos.append(idx)
            features_decision_pos.append(feature_decision_pos)

        self.feature_matrix = feature_matrix
        self.label_matrix = label_matrix
        self.features_decision_pos = features_decision_pos
        self.argsort_matrix = argsort_matrix.astype('int32')

    def decision_stump(self, weights):
        """
        根据权重计算决策桩，单层决策树
        :param weights: shape=(instances_num,) 样本对应权重
        :return:
        """
        min_error_rate = 1
        decision_feature_index = -1
        decision_feature_value = 0
        decision_type = '<'  # '<', '>'
        decision_value = 1  # 即满足decision条件的样本值为1

        # weight_matrix = self.label_matrix * weights
        for feature_index in range(self.instances_dim):
            # 该feature维度的样本权重
            # 此处出问题了。 self.label_matrix 按特征有序。 weights 原始样本序。
            feature_weight = self.label_matrix[feature_index] * weights[self.argsort_matrix[feature_index]]

            features_decision_pos = self.features_decision_pos[feature_index]
            for reverse_index in features_decision_pos:

                # 左边的错误率
                left_error = abs(feature_weight[:reverse_index][(self.label_matrix[feature_index] == -1)[:reverse_index]]).sum()
                # 右边的错误率
                right_error = abs(feature_weight[reverse_index:][(self.label_matrix[feature_index] == 1)[reverse_index:]]).sum()
                # 左预测1，右预测-1
                less_error = abs(left_error) + abs(right_error)
                # 左预测-1，右预测1
                greater_error = 1 - less_error

                if less_error < min_error_rate:
                    decision_feature_index = feature_index
                    decision_feature_value = (self.feature_matrix[feature_index][reverse_index]+self.feature_matrix[feature_index][reverse_index-1])/2
                    decision_type = '<'
                    min_error_rate = less_error
                if greater_error < min_error_rate:
                    decision_feature_index = feature_index
                    decision_feature_value = (self.feature_matrix[feature_index][reverse_index]+self.feature_matrix[feature_index][reverse_index-1])/2
                    decision_type = '>'
                    min_error_rate = greater_error

        return {'decision_feature_index': decision_feature_index,
                'decision_feature_value': decision_feature_value,
                'decision_type': decision_type,
                'min_error_rate': min_error_rate,
                'decision_value': decision_value
                }


def iris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    # 转换为 1/-1 的二分类数据
    data_index = y != 1
    X = X[data_index]
    y = y[data_index]-1
    # print(y)

    # split train test
    return train_test_split(
        X, y, test_size=0.33, random_state=1)


def cancer():
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = (breast_cancer.target * 2) - 1
    return train_test_split(
        X, y, test_size=0.33, random_state=42)


if __name__ == "__main__":

    # X_train, X_test, y_train, y_test = iris()
    # iris数据集什么情况？
    # (X_train[:, 2] > 3.2).astype('int') * 2 - 1 == y_train

    X_train, X_test, y_train, y_test = cancer()
    adaboost = AdaBoosting(verbose=1, n_estimators=40)
    adaboost.fit(X_train, y_train)
    # adaboost.fit(np.array([0,1,2,3,4,5,6,7,8,9])[::-1].reshape((10, 1)),
    #              np.array([1,1,1,-1,-1,-1,1,1,1,-1])[::-1])
    # adaboost.fit(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((10, 1)),
    #              np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1]))

    error_rate = adaboost.error_rate(X_test, y_test)
    print("score: ", 1 - error_rate)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
    # lr_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=40)
    bdt.fit(X_train, y_train)
    print(bdt.score(X_test, y_test))
