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

import scipy.spatial.distance as distance


def read_corpus(corpus_file_name, total_words=100000):
    lines = open(corpus_file_name).readlines()
    lines = ' '.join(map(lambda x: x.strip(), lines))
    words = [word for word in lines.split(' ') if len(word)>0][:total_words]
    return words


def statistic_corpus(text_words, min_df=3):
    counts = Counter(text_words)
    words = [item[0] for item in counts.items() if item[1] >= min_df]
    word2id = dict(zip(words, range(len(words))))
    text_words = [word for word in text_words if word in word2id]
    return words, word2id, text_words


def get_train_vec(samples):
    global id2word, word2id, vocabulary_size
    batch_size = len(samples)
    y = np.zeros((batch_size, vocabulary_size))
    x = np.zeros((batch_size, vocabulary_size))

    for sample_idx in range(batch_size):
        y[sample_idx][word2id[samples[sample_idx][0]]] = 1
        for word in samples[sample_idx][1]:
            x[sample_idx][word2id[word]] = 1
    return x, y


def gene_train_corpus(text_words, window=2, batch_size=10):
    global target_idx
    # window指单侧宽度
    word_num = len(text_words)
    samples = []
    while True:
        target_idx += 1
        if target_idx >= word_num - window:
            target_idx = 0 + window
        if len(samples) < batch_size:
            sample = (text_words[target_idx], [])
            [sample[1].append(text_words[target_idx-window + i]) for i in range(window)]
            [sample[1].append(text_words[target_idx+1+i]) for i in range(window)]
            samples.append(sample)
            # print(sample)
        else:
            yield get_train_vec(samples)
            samples = []


def cos_sim(vec, vecs):
    """Takes 2 vectors a, b and returns the cosine similarity according
    to the definition of the dot product
    """

    dot_product = np.dot(vecs, vec)
    norm_a = np.linalg.norm(vec)
    norm_b = np.linalg.norm(vecs, axis=1)
    return dot_product / (norm_a * norm_b)


def get_similar(word, W):
    global id2word, word2id
    word_vec = W[word2id[word]]
    sims = cos_sim(word_vec, W)
    top10_ids = np.argsort(sims)[::-1][:11]
    print("%s top10:" % word)
    for id in top10_ids[1:]:
        print("%s %.4f" % (id2word[id], sims[id]))
    print()


if __name__ == "__main__":
    np.random.seed(1)
    # 1、语料
    # corpus_file_name = '../../../data_on_git/trainCorpus.txt_utf8'
    corpus_file_name = 'test_word2vec_corpus.txt'
    origin_text_words = read_corpus(corpus_file_name, 100000)

    # text_words，剔除低频词后的文章
    id2word, word2id, text_words = statistic_corpus(origin_text_words, min_df=3)
    vocabulary_size = len(id2word)

    # 2、生成训练语料
    target_idx = 3
    x, y = gene_train_corpus(text_words).__next__()

    bp = BackPropagation(structure=(vocabulary_size, 10, vocabulary_size), learning_rate=0.01)
    iter = 10000
    for i in range(iter):
        batch_x, batch_y = gene_train_corpus(text_words).__next__()
        bp.train_batch(batch_x, batch_y, iter=i)

    # word = '祖国'
    W = bp.W[0]
    get_similar('发展', W)
    get_similar('提升', W)
    get_similar('变大', W)
    get_similar('增强', W)


    # gensim
    sentences = word2vec.LineSentence(corpus_file_name)
    w2v_model = word2vec.Word2Vec(sentences, min_count=3, window=5, size=10)
    #w2v_model = Word2Vec(origin_text_words, size=10, window=5, min_count=3, workers=4)

    w2v_model.wv.similar_by_word('发展')
    w2v_model.wv.similar_by_word('提升')
    w2v_model.wv.similar_by_word('变大')
    w2v_model.wv.similar_by_word('增强')
