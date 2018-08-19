# encoding: utf-8
'''
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: lof.py
@time: 18-8-14 下午10:22
@desc:
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

# 构造训练样本
n_samples = 200  # 样本总数
outliers_fraction = 0.15  # 异常样本比例
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)


# rng = np.random.RandomState(42)
np.random.seed(5)
X = 0.3 * np.random.randn(n_inliers // 2, 2)
X_train = np.r_[X + 2, X - 2]  # 正常样本
X_train = np.r_[X_train, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]  # 正常样本加上异常样本

# fit the model
# clf = LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction)
# y_pred = clf.fit_predict(X_train)
# scores_pred = clf.negative_outlier_factor_
from LocalOutlierFactor import LocalOutlierFactor
lof = LocalOutlierFactor(instances=X_train, k=6)
scores_pred = lof.local_outlier_factor
# threshold = stats.scoreatpercentile(scores_pred, 100 * (1-outliers_fraction))  # 根据异常样本比例，得到阈值，用于绘图

threshold = 1
x_norm, y_norm = zip(*X_train[scores_pred<=threshold])
x_anormal, y_anormal = zip(*X_train[scores_pred>threshold])
plt.scatter(x_norm, y_norm, 20, color="#0000FF")
plt.scatter(x_anormal, y_anormal, 20, color="red")
plt.show()
#
# x, y = zip(*X_train)
#
# plt.scatter(x, y, 20, color="#0000FF")
#
# for instance in [[0, 0], [5, 5], [10, 10], [-8, -8]]:
#     # value = lof.local_outlier_factor(3, instance)
#     value = 2
#     color = "#FF0000" if value > 1 else "#00FF00"
#     plt.scatter(instance[0], instance[1], color=color, s=(value - 1) ** 2 * 10 + 20)
# plt.show()


#
#
# # plot the level sets of the decision function
# xx, yy = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))
# Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])  # 类似scores_pred的值，值越小越有可能是异常点
# Z = Z.reshape(xx.shape)
#
# plt.title("Local Outlier Factor (LOF)")
# # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)  # 绘制异常点区域，值从最小的到阈值的那部分
# a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')  # 绘制异常点区域和正常点区域的边界
# plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='palevioletred')  # 绘制正常点区域，值从阈值到最大的那部分
#
# b = plt.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white',
#                 s=20, edgecolor='k')
# c = plt.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black',
#                 s=20, edgecolor='k')
# plt.axis('tight')
# plt.xlim((-7, 7))
# plt.ylim((-7, 7))
# plt.legend([a.collections[0], b, c],
#            ['learned decision function', 'true inliers', 'true outliers'],
#            loc="upper left")
# plt.show()