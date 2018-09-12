# encoding: utf-8
'''
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: LocalOutlierFactor.py
@time: 18-8-14 下午10:34
@desc:  参考 https://blog.csdn.net/wangyibo0201/article/details/51705966
'''
from matplotlib import pyplot as plt
import numpy as np
from util.distance import euclidean
import heapq
from itertools import combinations, combinations_with_replacement


class LocalOutlierFactor(object):
    def __init__(self, instances, k=3, is_normalize=False, distance_function='euclidean'):
        self.instances = instances
        self.instance_num = len(instances)
        if k > self.instance_num-1:
            raise Exception('k不能超过样本数减一')
        else:
            self.k = k

        if distance_function == 'euclidean':
            self.distance_function = euclidean
        # self.is_normalize = is_normalize

    def run(self):
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
        self.the_k_distance = np.zeros(self.instance_num)

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
            result = np.array(heapq.nsmallest(self.k+1, for_get_topk, key=lambda x: x[0]))
            nearest_k_index = result[:, 1][1:].astype(np.int32)
            the_k_distance = all_distance[nearest_k_index[-1]]

            self.the_k_distance[index] = the_k_distance
            neighbours_idx.append(nearest_k_index)
        self.neighbours_idx = np.array(neighbours_idx)

    def _compute_reach_distance(self):
        """
        计算所有实例的k之间互相的k可达距离
        reach_distance_2D[1,2] 表示 点1到点2的k可达距离。 max(k_distince(1), distance(1,2))

        :return:
        """
        # 二维数组存储
        self.reach_distance_2D = np.zeros([self.instance_num, self.instance_num])
        for row_idx in range(self.instance_num):
            for col_idx in range(self.instance_num):
                self.reach_distance_2D[row_idx][col_idx] = max(self.the_k_distance[row_idx],
                                                               self.distance_function(self.instances[col_idx], self.instances[row_idx]))

    def _compute_local_reachability_density(self):
        """
        局部可达密度
        :return:
        """
        self.local_reachability_density = np.zeros(self.instance_num)

        for instance_index in range(self.instance_num):
            sum_neighbours_reach_distance = 0
            for neighbour_idx in self.neighbours_idx[instance_index]:
                sum_neighbours_reach_distance += self.reach_distance_2D[neighbour_idx][instance_index]
            self.local_reachability_density[instance_index] = self.k / sum_neighbours_reach_distance

    def _compute_local_outlier_factor(self):
        """
        计算离群点   该点的局部可达密度 大于 其邻域点的局部可达密度平均  则该点为离群点， 可定义阈值
        :return:
        """
        self.local_outlier_factor = np.zeros(self.instance_num)
        for instance_index in range(self.instance_num):
            lrd_instance = self.local_reachability_density[instance_index]
            sum_lrd_neighbour = 0
            for neighbour_idx in self.neighbours_idx[instance_index]:
                sum_lrd_neighbour = self.local_reachability_density[neighbour_idx]
            self.local_outlier_factor[instance_index] = sum_lrd_neighbour / (self.k * lrd_instance)


if __name__=="__main__":
    instances = np.array([
        (-4.8447532242074978, -5.6869538132901658),
        (1.7265577109364076, -2.5446963280374302),
        (-1.9885982441038819, 1.705719643962865),
        (-1.999050026772494, -4.0367551415711844),
        (-2.0550860126898964, -3.6247409893236426),
        (-1.4456945632547327, -3.7669258809535102),
        (-4.6676062022635554, 1.4925324371089148),
        (-3.6526420667796877, -3.5582661345085662),
        (6.4551493172954029, -0.45434966683144573),
        (-0.56730591589443669, -5.5859532963153349),
        (-5.1400897823762239, -1.3359248994019064),
        (5.2586932439960243, 0.032431285797532586),
        (6.3610915734502838, -0.99059648246991894),
        (-0.31086913190231447, -2.8352818694180644),
        (1.2288582719783967, -1.1362795178325829),
        (-0.17986204466346614, -0.32813130288006365),
        (2.2532002509929216, -0.5142311840491649),
        (-0.75397166138399296, 2.2465141276038754),
        (1.9382517648161239, -1.7276112460593251),
        (1.6809250808549676, -2.3433636210337503),
        (0.68466572523884783, 1.4374914487477481),
        (2.0032364431791514, -2.9191062023123635),
        (-1.7565895138024741, 0.96995712544043267),
        (3.3809644295064505, 6.7497121359292684),
        (-4.2764152718650896, 5.6551328734397766),
        (-3.6347215445083019, -0.85149861984875741),
        (-5.6249411288060385, -3.9251965527768755),
        (4.6033708001912093, 1.3375110154658127),
        (-0.685421751407983, -0.73115552984211407),
        (-2.3744241805625044, 1.3443896265777866),
        (-10,-10),
        (15,15),
        (20,20)
    ])

    lof = LocalOutlierFactor(instances=instances, k=3)
    scores_pred = lof.run()

    threshold = 0.8
    x_norm, y_norm = zip(*instances[scores_pred <= threshold])
    x_anormal, y_anormal = zip(*instances[scores_pred > threshold])
    plt.scatter(x_norm, y_norm, 20, color="#0000FF")
    plt.scatter(x_anormal, y_anormal, 20, color="red")
    plt.show()


    # x, y = zip(*instances)
    # plt.scatter(x, y, 20, color="#0000FF")
    #
    # for instance in [[0, 0], [5, 5], [10, 10], [-8, -8]]:
    #     # value = lof.local_outlier_factor(3, instance)
    #     value = 2
    #     color = "#FF0000" if value > 1 else "#00FF00"
    #     plt.scatter(instance[0], instance[1], color=color, s=(value - 1) ** 2 * 10 + 20)
    # plt.show()
