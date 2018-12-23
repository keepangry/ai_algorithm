#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-23 下午1:57
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : hmm.py
# @Software: PyCharm
# 隐马尔科夫模型  矩阵方式计算
import numpy as np


class Hmm(object):
    def __init__(self):
        # 观察序列 [1, 2, 0]
        self.observe_seqs = None

        # 序列长度
        self.N = None

        # 隐含状态个数
        self.M = None

        # 不同观测个数
        self.U = None

        # alpha 存储前向计算时 各时刻的各隐含状态累积概率
        # N * M   N序列长度，M隐含状态个数
        self.alpha = None

        # beta 存储后向计算 各状态累积概率
        self.beta = None

        self.trans_prob = None
        self.emmission_prob = None
        self.start_prob = None

    def predict(self, sentence):
        return self.viterbi(sentence)

    # viterbi
    def viterbi(self, sentence):
        self.N = len(sentence)
        self.observe_seqs = []
        for w in sentence:
            # 未登录字处理! TODO 暂时设置为0，及observes为0的字
            if w not in self.observes:
                self.observe_seqs.append(0)
            else:
                self.observe_seqs.append(self.observes[w])

        # 存放最大的概率
        V = np.mat(np.zeros((self.N, self.M)))

        # 使当前状态最大概率对应的上一状态
        P = np.mat(np.zeros((self.N, self.M)))
        for t in range(self.N):
            if t == 0:
                V[t] = np.multiply(self.start_prob, self.emmission_prob[:, self.observe_seqs[t]].T)
            else:
                tmp = np.multiply(np.multiply(self.trans_prob, self.emmission_prob[:, self.observe_seqs[t]].T),
                                  V[t - 1].T)
                V[t] = np.max(tmp, axis=0)
                P[t - 1] = np.argmax(tmp, axis=0)

        # 从V和P中找最佳序列
        end_state = np.argmax(V[-1])
        seq = [end_state]
        # 从P中倒推寻找最佳路径
        for i in range(self.N - 1):
            idx = self.N - 2 - i
            end_state = int(P[idx, end_state])
            seq += [end_state]
        seq.reverse()

        # 解码出隐含状态
        n_2_state = dict(zip(hmm.states.values(), hmm.states.keys()))
        result = []
        for i in range(self.N):
            result.append((sentence[i], n_2_state[seq[i]]))
        return result

    # 有标注的参数学习
    # 后续可以用这个学一下分词。
    def params_compute(self, corpus):
        """
        corpus = [
                [('shop', 'Rainy'), ('clean', 'Rainy'), ('shop', 'Sunny'), ('walk', 'Rainy'), ('shop', 'Sunny'),
                 ('clean', 'Sunny')],
                [('walk', 'Sunny'), ('shop', 'Rainy'), ('shop', 'Rainy'), ('walk', 'Sunny'), ('clean', 'Sunny'),
                 ('walk', 'Rainy')],
                [('clean', 'Rainy'), ('clean', 'Sunny'), ('walk', 'Sunny'), ('shop', 'Rainy'), ('shop', 'Rainy'),
                 ('clean', 'Sunny')],
                [('clean', 'Rainy'), ('walk', 'Sunny'), ('clean', 'Sunny'), ('shop', 'Sunny'), ('walk', 'Sunny'),
                 ('clean', 'Rainy')],
                [('shop', 'Rainy'), ('clean', 'Rainy'), ('walk', 'Rainy'), ('clean', 'Rainy'), ('shop', 'Rainy'),
                 ('walk', 'Sunny')],
            ]
        :param corpus:
        :return:
        """
        # 需要有隐含状态标注的语料， 学习hmm的参数
        # ['shop', 'clean', 'walk', 'shop']
        # Rainy , 'Sunny
        # 标注语料， 多个序列， （观测， 隐状态）

        # 状态转移计数
        A = np.mat(np.zeros((self.M, self.M)))
        # 隐状态到观测计数
        B = np.mat(np.zeros((self.M, self.U)))
        # 初始状态计数
        PI = np.mat(np.zeros(self.M))

        for seqs in corpus:
            pre_seq = None
            for seq in seqs:
                state = seq[1]
                observe = seq[0]
                B[self.states[state], self.observes[observe]] += 1

                # 不为None时才能计算A值
                if pre_seq is not None:
                    pre_state = pre_seq[1]
                    A[self.states[pre_state], self.states[state]] += 1
                else:
                    PI[0, self.states[state]] += 1
                pre_seq = seq

        self.trans_prob = A / np.sum(A, axis=1)
        self.emmission_prob = B / np.sum(B, axis=1)
        self.start_prob = PI / np.sum(PI)

    def fit(self, corpus):
        states = {}
        state_idx = 0
        observes = {}
        observe_idx = 0
        for seqs in corpus:
            for seq in seqs:
                if seq[0] not in observes:
                    observes[seq[0]] = observe_idx
                    observe_idx += 1
                if seq[1] not in states:
                    states[seq[1]] = state_idx
                    state_idx += 1
        self.states = states
        self.observes = observes
        self.M = len(self.states)
        self.U = len(self.observes)

        # 更新参数
        self.params_compute(corpus)

    # TODO: Baum-Welch 无标注学习
    # 该算法比较难实现
    def Baum_Welch(self):
        # 1、根据   计算 γ gamma ξ  xi
        # gamma = np.mat(np.zeros((self.N, self.M)))
        A = np.multiply(self.alpha, self.beta)
        b = np.sum(A, axis=1)
        gamma = A / b

        # 这个不知道该怎么用矩阵计算
        xi = np.mat(np.zeros((self.N, self.M, self.M)))
        print(gamma)


def gene_corpus(file_path):
    # B M E S 隐含状态
    corpus = []
    with open(file_path) as fr:
        for line in fr:
            corpus_seq = []
            line = line.strip().split(' ')
            for word in line:
                if len(word) == 1:
                    corpus_seq.append((word[0], 'S'))
                if len(word) >= 2:
                    corpus_seq.append((word[0], 'B'))
                    for w in word[1:-1]:
                        corpus_seq.append((w, 'M'))
                    corpus_seq.append((word[-1], 'E'))
            corpus.append(corpus_seq)
    return corpus


def label_to_sentence(viter):
    result = ""
    for item in viter:
        result += item[0]
        if item[1] == 'S' or item[1] == 'E':
            result += ' '
    return result.strip()


if __name__ == '__main__':
    hmm = Hmm()
    # test
    # corpus = [
    #     [('shop', 'Rainy'), ('clean', 'Rainy'), ('shop', 'Sunny'), ('walk', 'Rainy'), ('shop', 'Sunny'),
    #      ('clean', 'Sunny')],
    #     [('walk', 'Sunny'), ('shop', 'Rainy'), ('shop', 'Rainy'), ('walk', 'Sunny'), ('clean', 'Sunny'),
    #      ('walk', 'Rainy')],
    #     [('clean', 'Rainy'), ('clean', 'Sunny'), ('walk', 'Sunny'), ('shop', 'Rainy'), ('shop', 'Rainy'),
    #      ('clean', 'Sunny')],
    #     [('clean', 'Rainy'), ('walk', 'Sunny'), ('clean', 'Sunny'), ('shop', 'Sunny'), ('walk', 'Sunny'),
    #      ('clean', 'Rainy')],
    #     [('shop', 'Rainy'), ('clean', 'Rainy'), ('walk', 'Rainy'), ('clean', 'Rainy'), ('shop', 'Rainy'),
    #      ('walk', 'Sunny')],
    # ]
    # hmm.hmm_train(corpus)

    # 分词
    file_path = '/home/yangsen/workspace/ai_algorithm/data_on_git/trainCorpus.txt_utf8'
    corpus = gene_corpus(file_path)
    hmm.fit(corpus)

    print(label_to_sentence(hmm.predict(sentence=list("新华网驻东京记者报道"))))
    # 新华网 驻 东京 记者 报道

    print(label_to_sentence(hmm.predict(sentence=list("城乡经济体制改革向纵深稳步发展"))))
    # 国际 形势 既 有 令人 欣慰 的 新 发展

    print(label_to_sentence(hmm.predict(
        sentence=list("人民网北京5月15日电 据中央纪委监察部网站消息，近日，中国石化实施的公务用车改革统计汇总显示：全系统共减少各类公务用车5820辆、司勤人员4287人，公车费用大幅下降。"))))
    # 人民 网 北京 5 月 1 5 日电   据 中央 纪委 监察 部网 站 消息 ， 近日 ， 中国 石化 实施 的 公务 用车 改革 统计汇 总 显示 ： 全系 统共 减少 各类 公务 用车 5 8 2 0 辆 、 司勤 人员 4 2 8 7 人 ， 公车 费用 大幅 下降 。
