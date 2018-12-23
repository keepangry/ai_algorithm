# encoding: utf-8
'''
@author: yangsen
@license:
@contact: 0@keepangry.com
@software:
@file: KDTree.py
@time: 18-8-30 下午9:11
@desc:
'''

import numpy as np
from util.distance import euclidean

class Node(object):
    def __init__(self, instance_index=None,split_feature_index=None,left=None,right=None):
        self.left = left
        self.right = right
        self.instance_index = instance_index
        self.split_feature_index = split_feature_index




class KDTree(object):
    def __init__(self):
        pass

    def fit(self, instances):
        self.instances = instances
        self.instance_num = instances.shape[0]
        self.dimension_num = instances.shape[1]

        self.instance_indexes = np.arange(self.instance_num)

        # 创建树
        self.kdtree = self._create_struct(self.instance_indexes)


    def _create_struct(self, instance_indexes):
        if instance_indexes.shape[0] <= 0 :
            return None

        curr_instances = self.instances[instance_indexes]
        # 计算各维度方差
        variance = np.var(curr_instances, axis=0)

        # 找到方差最大的维度
        split_feature_index = variance.argmax()

        # 找该维度下的中值点
        split_feature = curr_instances[:,split_feature_index]

        # 中值可能重复，但至少有一个
        # 此处太恶心，如果是偶数，numpy会取两偶数的均值，且未找到不取均值的方法。
        # median = np.median(split_feature)
        tmp_split_feature = list(split_feature)
        if len(tmp_split_feature)%2==0:
            tmp_split_feature.pop()
            median = np.median(tmp_split_feature)
        else: # 奇数个
            median = np.median(split_feature)

        median_index = np.argwhere(split_feature == median)[0][0]
        instance_index = instance_indexes[ median_index ]

        root = Node(instance_index, split_feature_index)

        # 先删除当前节点 样本内索引，不然递归出现重复
        left_instance_select = split_feature <= median
        left_instance_select[median_index] = False
        left_instance_indexes = instance_indexes[left_instance_select]
        right_instance_indexes = instance_indexes[ split_feature > median ]

        root.left = self._create_struct( left_instance_indexes )
        root.right = self._create_struct( right_instance_indexes )
        return root

    ##
    def find_dist_node(self, instance, dist):
        """
        KD树可以有几种查找方式：1、找最近的一个点  2、找k近邻   3、找小于某距离的所有点
        对于1和2，在较低维时，效率很好，但是高维会出现问题，因为在判断时只能看到一维的距离，而判断1维距离小于欧式距离便会进入子树去探索。
        往往一维距离是小于欧式距离的，这导致探索非常多的点。

        对于3，更是需要维度低。感觉对于3，该方法并不能提升太多效率，除非某距离很小。



        TODO: 所以，目前看，离该点小于某距离的所有临近点，会有较高的复杂度。需要遍历很多值啊
        :param instance:
        :param dist:
        :return:
        """

        curr_root = self.kdtree
        curr_instance = self.instances[curr_root.instance_index]
        node_stack = []
        dist_instance = []

        # 从跟开始，查到叶子，遇到小于指定距离的，加入的栈中
        while curr_root:
            node_stack.append(curr_root)
            curr_instance = self.instances[curr_root.instance_index]
            curr_dist = euclidean(curr_instance, instance)
            if curr_dist <= dist:
                dist_instance.append(curr_instance)

            # 值小于节点值，向左子树遍历
            if instance[curr_root.split_feature_index] <= curr_instance[curr_root.split_feature_index]:
                curr_root = curr_root.left
            else:
                curr_root = curr_root.right

        # 回溯查找。
        # 越靠近最后的，离样本越近。
        # 如果单维度距离，小于指定距离，则进行左右查找，继续靠近，路上遇到的，凡是小于指定距离的均填入
        while node_stack:
            back_root = node_stack.pop()
            curr_instance = self.instances[back_root.instance_index]
            if abs(curr_instance[back_root.split_feature_index] - instance[back_root.split_feature_index]) <= dist:
                if instance[back_root.split_feature_index] <= curr_instance[back_root.split_feature_index]:
                    curr_root = back_root.left
                else:
                    curr_root = back_root.right

                if curr_root:
                    node_stack.append(curr_root)
                    curr_instance = self.instances[curr_root.instance_index]
                    curr_dist = euclidean( curr_instance, instance )
                    if curr_dist <= dist:
                        dist_instance.append(curr_instance)

        return np.array(dist_instance)



if __name__ == "__main__":
    dimension_num = 2
    instance_num = 10

    # 构造不同维度不同的方差
    np.random.seed(1)
    instances = np.multiply(np.random.randn(10, 2), np.array([1, 2]))
    print(instances)

    kdtree = KDTree()
    kdtree.fit(instances)
    neighbor = kdtree.find_dist_node([1,2], 3)
    print( len(neighbor) )
    print( neighbor )
