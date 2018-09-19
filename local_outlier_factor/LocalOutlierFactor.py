"""
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: LocalOutlierFactor.py
@time: 18-8-14 下午10:34
@desc:  参考 https://blog.csdn.net/wangyibo0201/article/details/51705966
"""

from matplotlib import pyplot as plt
import numpy as np
from util.distance import euclidean
import heapq
from itertools import combinations, combinations_with_replacement

"""
1、第k距离：  距离某点第k近的点到该点的距离，设点为o，设第k距离为nearest_dis(o,num=k)。 第k距离有多个点，则全部包含。
2、第k距离邻域： 所有到该点距离小于等于nearest_dis(o,num=k) 的所有点
3、reach-distance可达距离： 点o到另一点p。则点o到点p的可达距离为：reach_dis(o, p) =  max( nearest_dis(o,num=k) , distance(o,p) )，即在邻域内的点视为相等，均为nearest_dis(o,num=k)
4、local reachability density局部可达密度： 设当前点为o,（第k邻域内点到o的平均可达距离的倒数）！！ 
      o_i为距离o第i近的点。   Σi~k  1 / reach_dis(o_i, o)
      o点邻域内的点，到o的可达距离越远则该值越小。
      局部可达密度 越大 代表该点距离邻域内的点的距离越近，可认为该点处于较密集的地方。
      越小，表明该点所在位置距其邻域越远。

5、local outlier factor局部离群因子：
    表示点o的 [邻域点N_k(o)的局部可达密度] 的平均数 与 [点o的局部可达密度](4)  的比 。 

    该值越高，说明，邻域点处于密集， 而该点处于较不密集。
    如果以1为阈值： 大于1，则说明该点所处的密度，小于其邻域点所处的密度的平均值。

"""


class LocalOutlierFactor(object):
    def __init__(self, instances, k=3, is_normalize=False, distance_function='euclidean'):
        self.instances = instances
        self.instance_num = len(instances)
        if k > self.instance_num - 1:
            raise Exception('k不能超过样本数减一')
        else:
            self.k = k

        if distance_function == 'euclidean':
            self.distance_function = euclidean
        # self.is_normalize = is_normalize

        # 样本用索引替代，保存该点的 第k距离
        self.the_k_distance = np.zeros(self.instance_num)

        # 二维数组。保存该点的邻域的 样本索引。
        self.neighbours_idx = np.zeros((self.instance_num, k))

        # 可达距离。   计算出所有点到所有点的可达距离。
        self.reach_distance_2D = np.zeros((self.instance_num, self.instance_num))

        # 局部可达密度。 该点的邻域点到该点平均可达距离的倒数
        self.local_reachability_density = np.zeros(self.instance_num)

        # 局部离群因子。 所有点的 离群因子(程度/得分)
        self.local_outlier_factor = np.zeros(self.instance_num)

    def fit(self):
        self._compute_k_distance()
        self._compute_reach_distance()
        self._compute_local_reachability_density()
        self._compute_local_outlier_factor()

        return self.local_outlier_factor

    def _compute_k_distance(self):
        """
        计算样本的k临域信息
        :return:
        """
        neighbours_idx = []
        for index in range(self.instance_num):
            all_distance = np.zeros(self.instance_num)
            for idx, _instance in enumerate(self.instances):
                all_distance[idx] = self.distance_function(self.instances[index], self.instances[idx])

            # TODO: 此处进行sort会增加复杂度，应改写。
            # 使用排序获取topk
            # all_distance_sorted_index = np.argsort(all_distance)
            # the_k_distance = all_distance[all_distance_sorted_index[self.k]]
            # nearest_k_index = all_distance_sorted_index[1:self.k+1]

            # 使用堆获取topk
            for_get_topk = np.vstack((all_distance, np.arange(len(all_distance)))).T
            result = np.array(heapq.nsmallest(self.k + 1, for_get_topk, key=lambda x: x[0]))

            nearest_k_index = result[:, 1][1:].astype(np.int32)

            the_k_distance = all_distance[nearest_k_index[-1]]

            self.the_k_distance[index] = the_k_distance
            neighbours_idx.append(nearest_k_index)
        self.neighbours_idx = neighbours_idx

    def _compute_reach_distance(self):
        """
        计算所有实例的k之间互相的k可达距离
        reach_distance_2D[1,2] 表示 点1到点2的k可达距离。 max(k_distince(1), distance(1,2))

        :return:
        """
        for row_idx in range(self.instance_num):
            for col_idx in range(self.instance_num):
                self.reach_distance_2D[row_idx][col_idx] = max(self.the_k_distance[row_idx],
                                                               self.distance_function(self.instances[row_idx],
                                                                                      self.instances[col_idx]))

    def _compute_local_reachability_density(self):
        """
        局部可达密度
        :return:
        """
        for instance_index in range(self.instance_num):
            sum_neighbours_reach_distance = 0
            for neighbour_idx in self.neighbours_idx[instance_index]:
                sum_neighbours_reach_distance += self.reach_distance_2D[neighbour_idx][instance_index]
            self.local_reachability_density[instance_index] = 1 / (sum_neighbours_reach_distance / self.k)

    def _compute_local_outlier_factor(self):
        """
        计算离群点   该点的局部可达密度 大于 其邻域点的局部可达密度平均  则该点为离群点， 可定义阈值
        :return:
        """
        for instance_index in range(self.instance_num):
            lrd_instance = self.local_reachability_density[instance_index]
            sum_lrd_neighbour = 0
            for neighbour_idx in self.neighbours_idx[instance_index]:
                sum_lrd_neighbour += self.local_reachability_density[neighbour_idx]
            self.local_outlier_factor[instance_index] = (sum_lrd_neighbour / self.k) / lrd_instance


if __name__ == "__main__":
    np.random.seed(1)
    x = 5 + 3 * np.random.randn(100)
    y = 5 + 2 * np.random.randn(100)
    instances = np.vstack((x, y)).T

    # extra = np.array([
    #     (-5, -5),
    #     # (15, 15),
    #     (14, 14),
    #     (20, 20),
    #     (18, 20),
    #     (20, 18),
    #     (18, 18)
    # ])
    #
    # instances = np.vstack((instances, extra))
    # instances = extra

    lof = LocalOutlierFactor(instances=instances, k=5)
    scores_pred = lof.fit()

    # threshold = 1
    threshold = 1.5
    x_norm = instances[scores_pred <= threshold][:, 0]
    y_norm = instances[scores_pred <= threshold][:, 1]
    plt.scatter(x_norm, y_norm, 20, color="#0000FF")

    if len(instances[scores_pred > threshold]) > 0:
        x_anormal = instances[scores_pred > threshold][:, 0]
        y_anormal = instances[scores_pred > threshold][:, 1]
        plt.scatter(x_anormal, y_anormal, 20, color="red")
    plt.show()
