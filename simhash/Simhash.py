#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/9/14 下午3:52
# @Author  : yangsen07@meituan.com
# @Site    : 
# @File    : Simhash.py
# @Software: PyCharm


import hashlib
import numpy as np


def hash(str):
    md5 = hashlib.md5(str.encode('utf-8')).hexdigest()
    hash_arr = np.zeros(64)
    for i in range(32):
        num = ord(md5[i]) % 4
        if num == 0:
            hash_arr[2*i] = 0
            hash_arr[2*i+1] = 0
        elif num == 1:
            hash_arr[2 * i] = 0
            hash_arr[2 * i + 1] = 1
        elif num == 2:
            hash_arr[2 * i] = 1
            hash_arr[2 * i + 1] = 0
        elif num == 3:
            hash_arr[2 * i] = 1
            hash_arr[2 * i + 1] = 1
    return hash_arr


class Simhash(object):
    def __init__(self, value_list):
        hash_sum = np.zeros(64)
        for value in value_list:
            hash_sum += hash(value)

        value_len = int(len(value_list) / 2)
        # 此处注意先后顺序
        hash_sum[hash_sum < value_len] = 0
        hash_sum[hash_sum >= value_len] = 1

        self.simhash = hash_sum

    def distance(self, other):
        return int(sum(abs(self.simhash - other.simhash)))


if __name__ == "__main__":
    print(hash('3213211'))
    print(Simhash(list("我爱中国天安门啊")).distance(Simhash(list("我爱天安门中国啊"))))
    print(Simhash(list("我爱中国天安门啊")).distance(Simhash(list("我爱天安门中国呐"))))
    print(Simhash(list("9月12日，中国传媒大学一位英语系00后新生，给学校写了一封信。信中，这名新生十分担心AI翻译未来会抢了她的饭碗。"))
          .distance(Simhash(list("9月12日，中国传媒大学一位英语系00后新生，给学校写了一封信。1信中，这名新生十分担心AI翻译未来会抢了她的饭碗。"))))
