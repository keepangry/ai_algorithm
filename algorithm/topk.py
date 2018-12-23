#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/12 下午3:03
# @Author  : 0@keepangry.com
# @Site    : 
# @File    : topk.py
# @Software: PyCharm
import heapq
import numpy as np


def topk_partition(arr, k):
    """
    输出的前k大，但不保证有序
    :param arr:
    :param k:
    :return:
    """
    def partition(num_list, left, right, k):
        flag = num_list[left]
        i = left
        j = right
        while i < j:
            # print(flag,i,j,num_list)
            if num_list[i] > flag:
                i += 1
            elif num_list[j] < flag:
                j -= 1
            else:
                if num_list[i] == num_list[j]:
                    j -= 1
                num_list[i], num_list[j] = num_list[j], num_list[i]
        # print(flag,num_list)
        if i < k:
            return partition(num_list, i + 1, right, k)
        if i > k:
            return partition(num_list, left, i - 1, k)
        return num_list[:k]

    return partition(arr[:], 0, len(arr) - 1, k)


if __name__ == "__main__":
    a = [3, 44, 341, 4214, 34, 21]
    print(topk_partition(a, k=3))

    print(heapq.nsmallest(3, a))

    print(np.array(heapq.nsmallest(3, np.array([
        [0, 3], [0, 2], [0, 5], [4, 0]
    ]), key=lambda x: x[1])))

    # 获取topk的索引。
    for_get_topk = np.vstack((a, np.arange(len(a)))).T
    print(for_get_topk)
    result = np.array(heapq.nsmallest(3, for_get_topk, key=lambda x: x[0]))
    print(result[:, 1])

    # 如和获取topk的索引

