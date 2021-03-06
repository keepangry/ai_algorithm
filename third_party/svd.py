#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/10 10:28 AM
# @Author  : yangsen
# @Site    :
# @File    : svd.py.py
# @Software: PyCharm
import sys
import numpy as np
import os


# 相似度计算函数
def cos_sim(x, y):
    """
    根据余弦计算x,y的相似度
    :param x: 1*n array
    :param y: 1*n array
    :return:
    """
    similarity = 0.5 + 0.5 * np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
    return similarity


# 协同过滤推荐算法
# 注意：1.相似度计算结果规范到0-1之间，2.计算最终得分要加权计算，不是相似度和评分乘积
# 3.np.sum如果不指定维度，会统计所有和
def recommend(user_index, grade_arr, topk=10):
    """
    基于商品相似度的协同推荐算法
    :param user_index:
    :param grade_arr: 用户对商品的评分矩阵
    :return: 推荐的top k商品
    """
    if user_index < 0:
        raise Exception("user_index must more than 0")
    if grade_arr is None or grade_arr.shape[0] == 0:
        raise Exception("there is no data in grade_arr")
    # 利用svd降维去噪
    u, sigma, vt = np.linalg.svd(grade_arr)
    sigma_sum = np.sum(sigma)
    sigma_partsum = 0
    # for k in range(len(sigma)):
    #     sigma_partsum+=sigma[k]
    #     if sigma_partsum*1.0/sigma_sum>0.9:
    #         break
    sigma = np.eye(4) * sigma[:4]
    product_arr = grade_arr.T.dot(u[:, :4]).dot(np.linalg.inv(sigma))

    # 计算商品相似度
    product_dot = product_arr.dot(product_arr.T)
    prodcut_model = np.mat(np.sqrt(np.sum(product_arr ** 2, axis=1)))
    product_models = prodcut_model.T * prodcut_model
    prodcut_sim = product_dot / product_models.A * 0.5 + 0.5  # 使求的相似度值在0-1范围内
    # print(prodcut_sim)

    # 找出用户没有评分的商品
    nograde = grade_arr[user_index] == 0
    if nograde.shape[0] == 0:
        raise Exception("you rate everything")
    nograde_index = np.nonzero(nograde)[0]
    grade_index = np.nonzero(~nograde)[0]

    # 计算无评分商品的推荐得分
    sel_prodcut_sim = prodcut_sim[nograde_index][:, grade_index]
    user_product_grade = grade_arr[user_index][grade_index]
    score = sel_prodcut_sim.dot(user_product_grade) / np.sum(sel_prodcut_sim[:, ], axis=1)

    # 推荐top k
    sort_index = np.argsort(-score)
    topk_index = sort_index[:topk]
    return zip(*(nograde_index[topk_index], score[topk_index]))


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


grad_arr = np.array(loadExData2())
result = recommend(1, grad_arr, 3)
