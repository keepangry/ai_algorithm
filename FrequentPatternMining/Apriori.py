#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 4:30 PM
# @Author  : 0@keepangry.com
# @Site    : 
# @File    : Apriori.py
# @Software: PyCharm

"""
Apriori频繁项挖掘
https://blog.csdn.net/baimafujinji/article/details/53456931
https://www.cnblogs.com/llhthinker/p/6719779.html

https://www.cnblogs.com/qwertWZ/p/4510857.html

1、想法。
在判断是否是 某一集合的子集时，可以使用位的方式。 判断[0,1,1,0,0]是不是[0,1,1,0,1]的子集效率在O(n)
而使用set集合判断，效率是多少，不知道。

2、frozenset

3、关联规则
对于每个频繁项集itemset，产生itemset的所有非空子集（这些非空子集一定是频繁项集）；
对于itemset的每个非空子集s,如果
"""
import math


class Apriori(object):

    def __init__(self, instances, min_support=0.3, min_confidence=0.5, delimiter=" "):
        self.instances_num = len(instances)
        ## TODO:需考虑排序效率，是否排序是一定需要的
        self.instances = self.instances_sorted_2D(data_1D=instances, delimiter=delimiter)
        self.frozenset_instances = [frozenset(instance) for instance in self.instances]

        self.instances_sorted = ""
        self.min_support = min_support
        self.min_confidence = min_confidence

    def instances_sorted_2D(selft, data_1D, delimiter=" "):
        """
        对样本进行化
        :param data_1D:
        :param delimiter:
        :return:
        """
        return [sorted(line.strip().split(delimiter)) for line in data_1D]

    def Lk_by_threshold(self, Ck):
        threshold = math.ceil(self.instances_num * self.min_support)
        return {key: value for key, value in Ck.items() if value >= threshold}

    def generate_candidates(self, Lk, k):
        # 只影响第一次
        if k == 0:
            result = {}
            for instance in self.frozenset_instances:
                for item in instance:
                    item = frozenset([item])
                    result.update({item: result.get(item, 0) + 1})
            return result


        # 排序key,合并成k+1长度
        itemsets_2d = []
        for key in Lk.keys():
            itemsets_2d.append(sorted(list(key)))

        # 生成k+1长度itemset
        pre_itemsets_Kplus1 = []

        itemsets_length = len(itemsets_2d)
        i = 0
        j = 1
        while i < itemsets_length-1:
            if itemsets_2d[i][:-1] == itemsets_2d[j][:-1]:
                # 如果当前itemset的前k-1个item等于下一个的，则进行组合成k+1的itemset
                pre_itemsets_Kplus1.append(itemsets_2d[i] + itemsets_2d[j][-1:])
                j += 1
                # j 走到最后了
                if j == itemsets_length:
                    i += 1
                    j = i+1
            else:
                i += 1
                j = i+1

        itemsets_Kplus1 = []
        # 预剪枝，k+1的k长度子集，是否在Lk中，不在则剔除
        for itemset in pre_itemsets_Kplus1:
            flag = True
            for i in range(len(itemset)):
                itemset_k = frozenset(itemset[:i] + itemset[i + 1:])
                if itemset_k not in Lk:
                    flag = False
            if flag:
                itemsets_Kplus1.append(frozenset(itemset))

        # 遍历样本计算数量
        C_Kplus1 = {}
        for itemset in itemsets_Kplus1:
            for instance in self.frozenset_instances:
                if itemset.issubset(instance):
                    C_Kplus1.update({itemset: C_Kplus1.get(itemset, 0) + 1})
        return C_Kplus1

    def fit(self):
        freq_itemset = []
        support_dict = {}
        C = []
        L = {}
        i = 0
        while True:
            C = self.generate_candidates(L, i)
            L = self.Lk_by_threshold(C)
            # 未能产生更长的频繁项
            if len(L) == 0:
                break
            support_dict.update(L)
            freq_itemset.append(L)
            i += 1

        self.freq_itemset = freq_itemset
        self.support_dict = support_dict
        return freq_itemset, support_dict

    def fit_study(self):
        """
        手写代码调试时，一步一步写
        :return:
        """
        C0 = []
        L0 = {}

        C1 = self.generate_candidates(L0, 0)
        print(C1)
        L1 = self.Lk_by_threshold(C1)
        print(L1)

        C2 = self.generate_candidates(L1, 1)
        print(C2)
        L2 = self.Lk_by_threshold(C2)
        print(L2)

        C3 = self.generate_candidates(L2, 2)
        print(C3)
        if len(C3) == 0:
            return

    def PowerSetsRecursive2(self, items, is_null_set=False):
        """
        获取一个列表所有子集
        注意结果会含有空集
        :return:
        """
        # the power set of the empty set has one element, the empty set
        result = [[]]
        for x in items:
            result.extend([subset + [x] for subset in result])

        result = [frozenset(x) for x in result]
        if is_null_set:
            return result
        else:
            return result[1:]

    def rule(self):
        """
        获取满足置信度阈值的关联规则
        :return:
        """
        rule_dict = {}
        sub_set_list = []

        for lenK_itemsets in self.freq_itemset:
            # lenK_itemsets  长度为1的，长度为2的
            for itemset in lenK_itemsets:  # 频繁项
                for subset in self.PowerSetsRecursive2(itemset):
                    if subset == itemset:
                        continue
                    confidence = round(self.support_dict[itemset] / self.support_dict[subset],3)
                    if confidence < self.min_confidence:
                        continue
                    rule = (subset, itemset - subset, confidence)
                    if rule not in rule_dict:
                        rule_dict.update({rule: 0})

        return rule_dict


if __name__ == "__main__":
    instances = [
        "啤酒 尿布 垃圾箱",
        "啤酒 尿布 手电筒 电风扇",
        "啤酒 白酒 百事可乐",
        "垃圾箱 百事可乐 卫生纸",
        "啤酒 尿布 水杯 卫生纸",
        "垃圾箱 手电筒 百事可乐 水杯",
        # "垃圾箱 手电筒 水杯 垃圾箱",
    ]

    apr = Apriori(instances=instances, min_support=0.2, min_confidence=0.6)
    freq_itemset, support_dict = apr.fit()
    print(support_dict)
    print(freq_itemset)
    rule_dict = apr.rule()
    [print(rule) for rule in rule_dict.keys()]
    """
    (frozenset({'啤酒'}), frozenset({'尿布'}), 0.75)
    买啤酒的有75%买尿布，所以给买啤酒的推荐尿布。
    
    (frozenset({'尿布'}), frozenset({'啤酒'}), 1.0)
    """
