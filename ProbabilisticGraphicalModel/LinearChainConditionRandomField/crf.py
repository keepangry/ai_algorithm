#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-13 下午4:24
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : crf.py
# @Software: PyCharm
"""
https://github.com/lancifollia/crf  把我整懵了

"""

from ProbabilisticGraphicalModel.LinearChainConditionRandomField.dev_dataset import train_data, status_feature
import numpy as np
from collections import defaultdict


# label 的开始与结束
LABEL_START = '|-'
LABEL_END = '-|'

# 边界外X
PADDING_XS = ['*', '|']

class CRF(object):
    def __init__(self, status_feature):
        self.status_feature = status_feature


    def fit(self, train_data):
        self.sentences = train_data
        self.sentence_num = len(self.sentences)
        self.col_num = len(self.sentences[0][0])

        # 基础数据解析
        self.train_data_parsing()

        # 参数
        self.weights = np.zeros(self.feature_num)

        # 生成训练集
        self.train_X = self._gene_train_X()

        # 势
        self.tables = self._gene_potential_table(self.train_X, self.weights)

        a = 1

    def _gene_train_X(self):
        XX = []
        for sentence_vec in self.sentences:
            sentence_X = []

            sentence_length = len(sentence_vec)
            for pos_idx, X in enumerate(sentence_vec):
                y = sentence_vec[pos_idx][-1]
                y_pre = sentence_vec[pos_idx - 1][-1] if pos_idx > 0 else LABEL_START

                feature_index = []
                for feature_poses in self.status_feature.values():
                    feature_tuple = self.get_feature_tuple(feature_poses, sentence_vec, pos_idx)
                    key = (y_pre, y, feature_tuple)
                    feature_index.append(self.feature2index[key])
                sentence_X.append(( y_pre, y, feature_index))
            XX.append(sentence_X)
        return XX

    def _get_pos_feature_indexes(self, sentence_vec, pos_idx):
        y = sentence_vec[pos_idx][-1]
        y_pre = sentence_vec[pos_idx - 1][-1] if pos_idx > 0 else LABEL_START

        indexes = []
        for feature_poses in self.status_feature.values():
            feature_tuple = self.get_feature_tuple(feature_poses, sentence_vec, pos_idx)
            key = (y_pre, y, feature_tuple)
            indexes.append(self.feature2index[key])
        return np.array(indexes)


    def _gene_potential_table(self, train_X, weights):
        tables = []
        for sentence_index, train_X_vec in enumerate(train_X):
            table = np.zeros((self.label_num, self.label_num))
            for pos_idx, X in enumerate(train_X_vec):
                feature_indexes = np.array(X[2])
                y_pre = X[0]
                y = X[1]
                y_pre_index = self.label2index[y_pre]
                y_index = self.label2index[y]

                score = weights[feature_indexes].sum()
                table[y_pre_index, y_index] += score

            table = np.exp(table)

            # start初始化为1。且根自己势为0。
            if sentence_index == 0:
                table[0 + 1:] = 0
            else:
                table[:, 0] = 0
                table[0, :] = 0
            tables.append(table)
        return tables


    def train_data_parsing(self):
        # 第一遍遍历
        labels = set()
        # (y_pre, y, feature_tuple)
        self.feature_counts = defaultdict(int)

        for sentence_vec in self.sentences:
            sentence_length = len(sentence_vec)
            for pos_idx, X in enumerate(sentence_vec):
                # vector = ('我', '1', .., 'B'),
                y = X[-1]
                y_pre = sentence_vec[pos_idx-1][-1] if pos_idx > 0 else LABEL_START
                labels.add(y)

                # 遍历每个特征
                for feature_poses in self.status_feature.values():
                    # feature_pos_desc: ((-1, 0), (0, 0),),
                    feature_tuple = self.get_feature_tuple(feature_poses, sentence_vec, pos_idx)
                    key = (y_pre, y, feature_tuple)
                    self.feature_counts[key] += 1

        self.features = self.feature_counts.keys()
        self.feature_num = len(self.features)
        self.feature2index = dict(zip(self.features, range(self.feature_num)))

        self.labels = [LABEL_START] + list(labels) + [LABEL_END]
        self.label_num = len(self.labels)
        self.label2index = dict(zip(self.labels, range(self.label_num)))


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
            relative_word = PADDING_XS[feature_col_idx] if relative_pos > sentence_length - 1 or relative_pos < 0 else \
                sentence_vec[relative_pos][feature_col_idx]
            feature = "%s_%s" % (feature_col_idx, relative_word)
            feature_tuple.append(feature)
        return tuple(feature_tuple)



if __name__ == "__main__":
    crf = CRF(status_feature)
    crf.fit(train_data)

