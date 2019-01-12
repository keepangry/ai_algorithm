#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-12 上午9:39
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : crf.py
# @Software: PyCharm
from ProbabilisticGraphicalModel.LinearChainConditionRandomField.dev_dataset import train_data, feature
import numpy as np


class CRF(object):
    def __init__(self, feature):
        self.origin_feature = feature

        # 如果该位置在边界外，使用该默认字符串
        self.padding_word = '_'

    def fit(self, train):
        self.sentences = train

        # 根据数据解析出具体特征
        self.feature = self.parse_data_feature()

        # 构建特征函数
        self.feature_num = len(self.feature)
        self.weight = np.random.rand(self.feature_num)



    def parse_data_feature(self):
        feature = {}
        feature_index = 0

        # 遍历句子
        for sentence in self.sentences:
            sentence_length = len(sentence)

            # 遍历每个位置
            for pos_idx, (word, label) in enumerate(sentence):
                # 遍历每个特征
                for feature_name, feature_poses in self.origin_feature.items():
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
        return feature


















if __name__ == "__main__":
    crf = CRF(feature)
    crf.fit(train_data)
