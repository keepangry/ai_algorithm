#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-22 上午11:54
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : MinHash.py
# @Software: PyCharm
import numpy as np


class MinHash(object):
    def __init__(self, sets, hash_num=10, random_state=0):
        """

        :param sets: 二维列表，每一项是去重后的item列表。
        """
        np.random.seed(random_state)

        self.hash_num = hash_num

        # 所有集合
        self.sets = sets
        self.set_num = len(self.sets)

        # 1、计算所有item，并进行索引
        flatten_set = []
        [flatten_set.extend(s) for s in sets]
        # 此处需进行排序，否则set出来每次运行顺序会变化
        self.items = sorted(list(set(flatten_set)))
        self.item_num = len(self.items)
        self.item_to_index = dict(zip(self.items, range(self.item_num)))
        print(self.item_to_index)

        # 2、自动生成多个hash函数。 hash函数目的是降低打乱顺序的成本，只需要一次hash计算就可以了。
        hash_funcs = self._gene_hash_funcs(num=self.hash_num, mod=self.item_num)

        # 3、计算数值矩阵
        item_set_mat = np.full((self.item_num, self.set_num), 0, dtype='int')
        item_hash_mat = np.full((self.item_num, self.hash_num), 0, dtype='int')
        # 根据集合元素 填充矩阵
        for set_index in range(self.set_num):
            for item in self.sets[set_index]:
                item_index = self.item_to_index[item]
                item_set_mat[item_index][set_index] = 1
        # 计算hash矩阵
        for item_index in range(self.item_num):
            for hash_index in range(self.hash_num):
                item_hash_mat[item_index][hash_index] = hash_funcs[hash_index](item_index)

        # 4、计算集合的最小hash指纹
        hash_set_mat = np.full((self.hash_num, self.set_num), np.inf, dtype="float32")  # 此时元素为float类型
        # 遍历所有的item，根据是否存在该set，把hash值写入果结果表中
        for item_index in range(self.item_num):
            for set_index in range(self.set_num):
                # item_set为1,则把所有hash值拿出来，如果更小则写入结果
                if item_set_mat[item_index][set_index] == 1:
                    item_min_hash_vec = item_hash_mat[item_index]
                    origin_set_min_hash_vec = hash_set_mat[:, set_index]
                    hash_set_mat[:, set_index] = np.min([item_min_hash_vec, origin_set_min_hash_vec], axis=0)
        self.hash_set_mat = hash_set_mat
        self.item_set_mat = item_set_mat
        self.item_hash_mat = item_hash_mat

    def min_hash_vector(self, set_index):
        return self.hash_set_mat[:, set_index]

    def similarity(self, set_index1, set_index2):
        compare = np.equal(self.hash_set_mat[:, set_index1], self.hash_set_mat[:, set_index2])
        return compare.astype('int').sum() / compare.shape[0]

    def _gene_hash_funcs(self, num=3, mod=10):
        """

        :param num:
        :param mod:
        :return:
        """
        hash_funcs = []
        for i in range(num):
            hash_funcs.append(lambda x: (int(np.random.rand()*mod) * x + 5 + int(np.random.rand()*num)) % mod)
        return hash_funcs


if __name__ == "__main__":
    s = [
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'e', 'f', 'g'],
        ['a', 'b', 'c', 'h'],
        ['a', 'd'],
    ]

    minhash = MinHash(hash_num=10, sets=s, random_state=2)
    print(minhash.min_hash_vector(0))
    print(minhash.min_hash_vector(1))
    print(minhash.similarity(0, 1))
    print(minhash.similarity(1, 3))

    # doc test
    s = [
        list(set("通过上面的方法处理过后，一篇文档可以用一个很小的签名矩阵来表示，节省下很多但是，还有一个问题没有解决，那就是如果有很多篇文档，那么如果要找出相似度很高的文档，")),
        list(set("通过上面的方法处理过后，一篇文档可以用一个很小的签名矩阵很多内存空间；但是，还有一个问题没有解决，那就是如果有很多篇文档，那么如果要找出相似度很高的文档，")),
        list(set("但是，还有一个问题没有解决，那就是如果有很多篇文档，那么如果要找出相似度很高的文档，其中一种办法就是先计算出所有文档的签名矩阵，然后依次两两比较签名矩")),
        list(set("希运算模拟N次随机行打乱，然后统计|h(S1)=h(S2)|，就有 P=|h(S1)=h(S2)| / N 了。有了上一章节的证明，我们就可以通过多次进行最小哈希运算，来构造新的特征向量")),
    ]

    minhash = MinHash(hash_num=32, sets=s, random_state=2)
    print(minhash.min_hash_vector(0))
    print(minhash.min_hash_vector(1))
    print(minhash.similarity(0, 1))
    print(minhash.similarity(0, 2))
    print(minhash.similarity(0, 3))
