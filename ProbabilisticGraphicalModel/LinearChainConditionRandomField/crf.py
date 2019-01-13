#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-12 上午9:39
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : crf.py
# @Software: PyCharm
from ProbabilisticGraphicalModel.LinearChainConditionRandomField.dev_dataset import train_data, status_feature
import numpy as np
from collections import defaultdict

"""
特征函数：1、状态特征，y依赖x  2、转移特征，目前只考虑y依赖与y-1

"""

# label 的开始与结束
LABEL_START = '|-'
LABEL_END = '-|'


class CRF(object):
    def __init__(self, status_feature):
        self.status_feature = status_feature

        # 如果该位置在边界外，使用该默认字符串
        self.padding_word = '_'

    def fit(self, train_data):
        self.sentences = train_data

        # 基础数据解析
        self.train_data_parsing()

        # 根据数据解析出具体特征
        self.gene_feature_functions()

        self.feature_num = len(self.feature_functions)
        self.weights = np.zeros(self.feature_num)

        # train





    def train_data_parsing(self):
        # 第一遍遍历
        labels = set()
        # (feature_col_idx,feature_value)
        features = set()

        for sentence in self.sentences:
            for pos_idx, vector in enumerate(sentence):
                sentence_length = len(sentence)
                # vector = ('我', '1', .., 'B'),
                label = vector[-1]
                labels.add(label)

                # 遍历每个特征
                for feature_poses in self.status_feature.values():
                    # feature_pos_desc: ((-1, 0), (0, 0),),
                    feature_tuple = self.get_feature_tuple(feature_poses, sentence, pos_idx)
                    features.add(tuple(feature_tuple))


        self.labels = [LABEL_START] + list(labels) + [LABEL_END]
        self.label_num = len(self.labels)
        self.label2index = dict(zip(self.labels, range(self.label_num)))
        # 转移矩阵，保存转移次数
        self.label_trans_matrix = np.zeros((self.label_num, self.label_num))

        # 状态特征
        self.features = list(features)
        self.feature_num = len(features)
        self.feature2index = dict(zip(self.features, range(self.feature_num)))
        self.feature_status_matrix = np.zeros((self.feature_num, self.label_num))



        # 第二遍遍历
        for sentence in self.sentences:
            sentence_length = len(sentence)
            for pos_idx, vector in enumerate(sentence):
                label = vector[-1]

                # 计算label转移矩阵
                if pos_idx == 0:
                    self.label_trans_matrix[self.label2index[LABEL_START]][self.label2index[label]] += 1
                else:
                    pre_label = sentence[pos_idx - 1][-1]
                    self.label_trans_matrix[self.label2index[pre_label]][self.label2index[label]] += 1
                if pos_idx == sentence_length-1:
                    self.label_trans_matrix[self.label2index[label]][self.label2index[LABEL_END]] += 1

                # 计算状态矩阵
                # 遍历每个特征
                for feature_poses in self.status_feature.values():
                    # feature_pos_desc: ((-1, 0), (0, 0),),

                    # 遍历每个特征的元素，有些特征不止一个位置。
                    feature_tuple = self.get_feature_tuple(feature_poses, sentence, pos_idx)
                    self.feature_status_matrix[self.feature2index[feature_tuple]][self.label2index[label]] += 1

        a = 1

    def get_feature_tuple(self, feature_poses, sentence_vec, pos_idx):
        """
        获取特征
        :param feature_poses:
        :param sentence_vec:
        :param pos_idx:
        :return:
        """
        feature_tuple = []
        sentence_length = len(sentence_vec)
        for feature_pos, feature_col_idx in feature_poses:
            # feature_pos: (-1, 0)  (相对位置, 特征列)
            relative_pos = pos_idx + feature_pos
            relative_word = self.padding_word if relative_pos > sentence_length - 1 or relative_pos < 0 else \
                sentence_vec[relative_pos][feature_col_idx]
            feature = "%s_%s" % (feature_col_idx, relative_word)
            feature_tuple.append(feature)
        return tuple(feature_tuple)

    def gene_feature_functions(self):
        """
        生成特征函数
        函数输入 (y_pre,y,x,feature_col_idx)
        函数返回 0/1

        :return:
        """
        def trans_feature_function(y_pre,y):
            if y_pre in self.label2index and y in self.label2index:
                return np.sign(self.label_trans_matrix[self.label2index[y_pre]][self.label2index][y])
            return 0

        def status_feature_function(y,feature_tuple):
            if feature_tuple in self.feature2index and y in self.label2index:
                return np.sign(self.feature_status_matrix[self.feature2index[feature_tuple]][self.label2index][y])
            return 0


        # 转移特征函数
        trans_functions = [lambda y, y_pre, feature_tuple: trans_feature_function(y, y_pre) for _ in self.labels for _ in self.labels]

        status_functions = [lambda y, y_pre, feature_tuple: status_feature_function(y, feature_tuple) for _ in self.features for _ in self.labels]

        self.feature_functions = trans_functions + status_functions






    def parse_data_feature(self):
        feature = {}
        feature_index = 0

        # 状态特征！！
        # 遍历句子
        for sentence in self.sentences:
            sentence_length = len(sentence)

            # 遍历每个位置
            for pos_idx, (word, label) in enumerate(sentence):
                # 遍历每个特征
                for feature_name, feature_poses in self.status_feature.items():
                    # print(feature_name, feature_poses)
                    feature_words = ""

                    # 遍历每个特征的元素，有些特征不止一个位置。
                    for feature_pos in feature_poses:
                        relative_pos = pos_idx + feature_pos
                        relative_word = self.padding_word if relative_pos > sentence_length-1 or relative_pos < 0 else sentence[relative_pos][0]
                        feature_words += relative_word

                    feature_key = (feature_name, feature_words, label)
                    if feature_key not in feature:
                        feature[feature_key] = feature_index
                    feature_index += 1

        # 转移特征！！


        return feature


















if __name__ == "__main__":
    crf = CRF(status_feature)
    crf.fit(train_data)
