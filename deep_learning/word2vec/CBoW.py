#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/25 5:08 PM
# @Author  : yangsen
# @Site    : 
# @File    : CBoW.py
# @Software: PyCharm

"""
最基本的word2vec
CBoW + SoftMax
使用自己手写的神经网络。

语料使用训练分词的新闻语料。
~/data_on_git/trainCorpus.txt_utf8
"""

from deep_learning.manual_neural_network.FC_classification_v2 import BackPropagation
from collections import Counter
import numpy as np
from gensim.models import Word2Vec, word2vec
from config import BASE_PATH
import os
from util.distance import cos_sim


def read_corpus(corpus_file_name, total_words=100000):
    lines = open(corpus_file_name).readlines()
    lines = ' '.join(map(lambda x: x.strip(), lines))
    words = [word for word in lines.split(' ') if len(word) > 0][:total_words]
    return words


class CBOW(object):

    def __init__(self, text_words, min_df=3, window=2, batch_size=10):
        self.min_df = min_df
        self.window = window
        self.batch_size = batch_size

        self.target_idx = 0 + window

        self.id2word, self.word2id, self.text_words = self.statistic_corpus(text_words)
        self.vocabulary_size = len(self.id2word)

    def statistic_corpus(self, text_words):
        counts = Counter(text_words)
        id2word = [item[0] for item in counts.items() if item[1] >= self.min_df]
        word2id = dict(zip(id2word, range(len(id2word))))
        text_words = [word for word in text_words if word in word2id]
        return id2word, word2id, text_words

    def get_train_vec(self, samples):
        batch_size = len(samples)
        y = np.zeros((batch_size, self.vocabulary_size))
        x = np.zeros((batch_size, self.vocabulary_size))

        for sample_idx in range(batch_size):
            y[sample_idx][self.word2id[samples[sample_idx][0]]] = 1
            for word in samples[sample_idx][1]:
                x[sample_idx][self.word2id[word]] = 1
        return x, y

    def gene_train_corpus(self):
        text_words = self.text_words
        # window指单侧宽度
        word_num = len(text_words)
        samples = []
        while True:
            self.target_idx += 1
            if self.target_idx >= word_num - self.window:
                self.target_idx = 0 + self.window
            if len(samples) < self.batch_size:
                sample = (text_words[self.target_idx], [])
                [sample[1].append(text_words[self.target_idx-self.window + i]) for i in range(self.window)]
                [sample[1].append(text_words[self.target_idx+1+i]) for i in range(self.window)]
                samples.append(sample)
            else:
                yield self.get_train_vec(samples)
                samples = []

    def get_similar(self, word):
        W = self.bp.W[0]
        word_vec = W[self.word2id[word]]
        sims = cos_sim(word_vec, W)
        top10_ids = np.argsort(sims)[::-1][:11]
        print("%s top10:" % word)
        for id in top10_ids[1:]:
            print("%s %.4f" % (self.id2word[id], sims[id]))
        print()

    def train(self, iter=10000):
        self.bp = BackPropagation(structure=(self.vocabulary_size, 10, self.vocabulary_size), learning_rate=0.01)
        for i in range(iter):
            batch_x, batch_y = self.gene_train_corpus().__next__()
            self.bp.train_batch(batch_x, batch_y, iter=i)


if __name__ == "__main__":
    np.random.seed(1)
    # 1、语料
    # corpus_file_name = os.path.join(BASE_PATH, 'data_on_git/trainCorpus.txt_utf8')
    corpus_file_name = os.path.join(BASE_PATH, 'deep_learning/word2vec/test_word2vec_corpus.txt')
    origin_text_words = read_corpus(corpus_file_name, 10000)

    cbow = CBOW(text_words=origin_text_words)
    cbow.train(iter=100000)

    # 2、生成训练语料

    # word = '祖国'
    cbow.get_similar('发展')
    cbow.get_similar('提升')
    cbow.get_similar('变大')
    cbow.get_similar('增强')

    # gensim
    sentences = word2vec.LineSentence(corpus_file_name)
    w2v_model = word2vec.Word2Vec(sentences, min_count=3, window=5, size=10)
    #w2v_model = Word2Vec(origin_text_words, size=10, window=5, min_count=3, workers=4)

    w2v_model.wv.similar_by_word('发展')
    w2v_model.wv.similar_by_word('提升')
    w2v_model.wv.similar_by_word('变大')
    w2v_model.wv.similar_by_word('增强')
