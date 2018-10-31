#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 4:30 PM
# @Author  : yangsen
# @Site    : 
# @File    : Apriori.py
# @Software: PyCharm

"""
Apriori频繁项挖掘
https://blog.csdn.net/baimafujinji/article/details/53456931
https://www.cnblogs.com/llhthinker/p/6719779.html
https://wizardforcel.gitbooks.io/dm-algo-top10/content/apriori.html


https://www.kdnuggets.com/news/2000/n13/23i.html
啤酒与尿布大概虚构的，It's still a nice example, of course.


来源：
The Apriori algorithm was proposed by Agrawal and Srikant in 1994.
FP-Growth算法是韩嘉炜等人在2000年提出的关联分析算法


支持度：
    关联规则A->B的支持度support=P(AB)，指的是事件A和事件B同时发生的概率。
    【蛋糕】->【鲜花】 = P(蛋糕,鲜花）  即【蛋糕,鲜花】 共同出现的概率，全集数量是一定的。
置信度：
    confidence(A->B):  P(B|A) = P(AB) / P(A)
    confidence(蛋糕->鲜花)
    P(鲜花|蛋糕) = P( 蛋糕,鲜花 ) / P( 蛋糕 )

一共20位顾客，鲜花5个人买，蛋糕4个人买，鲜花和蛋糕同时买3个人。
    支持度：3/20
    置信度（蛋糕->鲜花）： 3 / 4  买蛋糕的有75%买鲜花。


说明：
itemset 频繁项集
k 频繁项集的项数

Apriori定律1：如果一个集合是频繁项集，则它的所有子集都是频繁项集。
Apriori定律2：如果一个集合不是频繁项集，则它的所有超集都不是频繁项集。


一、初始化
1、生成k=1项的频繁项集，计算其频度。C_k,此时k=1
2、根据支持度阈值，进行剪枝。保留满足支持度的频繁项。 L_k = threshold( C_k )   【Apriori定律2】

二、循环计算，产出频繁项
while:
    1、根据Lk生成，生成 C_k+1。  【重点】***

    2、产生满足 L_k+1 = threshold( C_k+1 )     【Apriori定律2】

    3、length( L_k+1 ) == 0: break;


三、关联规则
对于每个频繁项集itemset，产生itemset的所有非空子集（这些非空子集一定是频繁项集）；  【Apriori定律1】

对于itemset的每个非空子集s，如果满足 freq(itemset) / freq(s) >= min_confidence
则输出规则： s => (itemset-s)

itemset=(鲜花、蛋糕)   3次
s = 鲜花              5次
freq(itemset) / freq(s) =  3 / 5  > 0.5
则产生规则： 鲜花 => 蛋糕



备注：
1、与frozenset， set(可变集合)与frozenset(不可变集合)。frozenset可作为dict的key，set不可以。


复杂度：


FPGrowth O(n) 两次遍历

"""
import math


class Apriori(object):

    def __init__(self, instances, min_support=0.3, min_confidence=0.5, delimiter=" "):
        self.instances_num = len(instances)

        ## TODO:需考虑排序效率，是否排序是一定需要的
        self.instances = self.instances_sorted_2D(data_1D=instances, delimiter=delimiter)

        self.frozenset_instances = [frozenset(instance) for instance in self.instances]

        self.min_support = min_support
        self.threshold = math.ceil(self.instances_num * self.min_support)
        print("threshold >= {}".format(self.threshold))
        self.min_confidence = min_confidence

    def instances_sorted_2D(selft, data_1D, delimiter=" "):
        """
        对样本进行化
        :param data_1D:
        :param delimiter:
        :return:
        """
        # return [sorted(line.strip().split(delimiter)) for line in data_1D]
        return [line.strip().split(delimiter) for line in data_1D]

    def Lk_by_threshold(self, Ck):
        return {key: value for key, value in Ck.items() if value >= self.threshold}

    def generate_candidates(self, Lk, k):
        # 只影响第一次初始化
        if k == 0:
            one_item_freq_dict = {}
            for instance in self.frozenset_instances:
                for item in instance:
                    item = frozenset([item])
                    one_item_freq_dict.update({item: one_item_freq_dict.get(item, 0) + 1})
            return one_item_freq_dict


        # 排序key,合并成k+1长度
        itemsets_2d = []
        for key in Lk.keys():
            itemsets_2d.append(sorted(list(key)))
        """itemsets_2d
        A,B
        A,C
        A,D
        B,C
        B,F
        ==> 从2项生成3项
        A,B,C
        A,B,D
        A,C,D
        B,C,F
        """

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
        """Lk
        A,B
        A,C
        A,D
        B,C
        B,F
        ==> 从2项生成3项
        A,B,C   =>(k-1)子集 A,B、A,C、B,C 均在 Lk中，保留
        A,B,D   =>(k-1)子集 B,D 不在 Lk中，放弃。
        A,C,D   =>(k-1)子集 C,D 不在 Lk中，放弃。
        B,C,F   =>(k-1)子集 C,F 不在 Lk中，放弃。
        最终产出：k=3的只有： A,B,C
        """
        for itemset in pre_itemsets_Kplus1:
            flag = True
            for i in range(len(itemset)):
                itemset_k = frozenset(itemset[:i] + itemset[i + 1:])
                if itemset_k not in Lk:
                    flag = False
            if flag:
                itemsets_Kplus1.append(frozenset(itemset))

        # 遍历样本计算数量。   此处复杂度：k*n
        C_Kplus1 = {}
        for itemset in itemsets_Kplus1:
            for instance in self.frozenset_instances:
                if itemset.issubset(instance):
                    C_Kplus1.update({itemset: C_Kplus1.get(itemset, 0) + 1})
        return C_Kplus1

    def fit(self):
        """
        freq_itemset分级的结果。
        [
            [ [A],[B],[C] ],
            [ [A,B],[A,C],[B,C] ],
            [ [A,B,C] ],
        ]
        :return:
        """
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
        输入[1,2,3]
        返回[
            [1],[2],[3],
            [1,2],[1,3],[2,3]
            []  # if is_null_set=true
            [1,2,3]
        ]
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

        # 此处复杂度，如果项数过长，会造成爆炸计算。 m为项数，其全部子集数为：2^n.
        for lenK_itemsets in self.freq_itemset:
            # lenK_itemsets  长度为1的，长度为2的
            for itemset in lenK_itemsets:  # 频繁项
                for subset in self.PowerSetsRecursive2(itemset):
                    if subset == itemset:
                        continue
                    confidence = round(self.support_dict[itemset] / self.support_dict[subset], 3)
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

    apr = Apriori(instances=instances, min_support=0.4, min_confidence=0.6)
    freq_itemset, support_dict = apr.fit()
    print("频繁项集： ", freq_itemset)
    print(support_dict)

    rule_dict = apr.rule()
    print("关联规则：")
    [print(rule) for rule in rule_dict.keys()]


    """
    (frozenset({'啤酒'}), frozenset({'尿布'}), 0.75)
    买啤酒的有75%买尿布，所以给买啤酒的推荐尿布。
    
    (frozenset({'尿布'}), frozenset({'啤酒'}), 1.0)
    """
