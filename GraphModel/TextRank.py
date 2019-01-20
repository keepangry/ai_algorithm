#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-20 下午8:42
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : TextRank.py
# @Software: PyCharm
import itertools


class Node(object):
    def __init__(self, word):
        self.word = word
        self.score = 1
        self.connections = set()
        self.connect_num = 0
        self.new_score = 0

    def add(self, node):
        self.connections.add(node)
        self.connect_num = len(self.connections)


class TextRank(object):

    def __init__(self, sentences):
        self.k = 3
        self.d = 0.85
        self.eta = 0.001

        self.sentences = sentences
        for sentence in self.sentences:
            self.rank(sentence)

    def rank(self, sentence):
        # 1、所有节点
        words = list(set(sentence))
        nodes = []
        for word in words:
            nodes.append(Node(word))
        word2node = dict(zip(words, nodes))

        sentence_length = len(sentence)
        # 使用窗口建立图
        for i in range(sentence_length-self.k):
            iter = itertools.combinations(sentence[i:i+self.k], 2)
            for item in iter:
                word2node[item[0]].add(word2node[item[1]])
                word2node[item[1]].add(word2node[item[0]])

        # 迭代计算，直到稳定
        stop_flag = False
        while True:
            if stop_flag:
                break

            stop_flag = True
            for node in nodes:
                # 该场景下不可能是孤立点，此处不再考虑

                # 计算节点得分
                relate_score = 0
                for connect_node in node.connections:
                    relate_score += 1/connect_node.connect_num * connect_node.score
                new_score = (1 - self.d) + self.d * relate_score
                bias = abs(new_score - node.score)
                if bias > self.eta:
                    stop_flag = False
                node.new_score = new_score

            # 更新score
            for node in nodes:
                node.score = node.new_score

        result = []
        for node in nodes:
            result.append((node.word, node.score))

        result = sorted(result, key=lambda x: x[1], reverse=True)
        for i in result:
            print(i)


if __name__ == "__main__":

    sentences = [
        ['有', '媒体', '曝光', '高圆圆', '和', '赵又廷', '现身', '台北', '桃园', '机场', '的', '照片'],
    ]
    TextRank(sentences)
