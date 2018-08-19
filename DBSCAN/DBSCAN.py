# encoding: utf-8
'''
@author: yangsen
@license: (C) Copyright 2013-2018, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software: garner
@file: DBSCAN.py
@time: 18-8-19 上午10:42
@desc:  参考: https://www.cnblogs.com/pinard/p/6208966.html
'''
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import queue




class DBSCAN(object):
    def __init__(self, instances, eps=0.5, min_samples=5, distance_function='euclidean'):
        self.instances = instances
        self.instance_num = len(instances)
        self.eps = eps
        self.min_samples = min_samples
        self.distance_function = self._euclidean

        # 初始化
        self.Ω = {}  # 核心对象
        self.k = 0   # 从1开始计算簇， 0类为异常点
        self.to_visit_instance = {idx:'' for idx in range(self.instance_num)}  # 未访问样本，key为样本序号
        self.C = {}  # 簇划分

        # 找出所有核心对象
        self._compute_core_instance()
        if len(self.Ω) != 0:
            self._compute_C()
        else:
            print("无任何核心对象")

    def predict(self):
        label = np.zeros(self.instance_num)
        for key,value in self.C.items():
            for instance_index in value:
                label[instance_index] = key
        return label


    def _compute_C(self):
        # 簇, 直到核心对象集合为空
        while len(self.Ω) > 0:
            self.k += 1  # 序号
            q = queue.Queue()
            core_instance_start = list(self.Ω.keys())[0]
            # 选择一个核心对象开始

            # curr_C = [core_instance_index]
            # del self.Ω[core_instance_index]  # 排除该核心对象

            q.put(core_instance_start)
            curr_C = {}

            while q.qsize() > 0:
                instance_index = q.get()

                # 如果是核心对象，则把其eps邻域所有点放入待遍历
                if instance_index in self.Ω and instance_index in self.to_visit_instance:
                    [q.put(eps_neighbour) for eps_neighbour in self.eps_neighbours[instance_index]]

                # 如果该点未访问，则更新到簇中，并从未访问中删除该点
                #  注意，该处删除访问遍历，应该在核心放入之后
                if instance_index in self.to_visit_instance:
                    curr_C.update({instance_index: ''})
                    self.to_visit_instance.pop(instance_index)

            # 保存簇信息
            self.C[self.k] = list(curr_C.keys())

            # 把该簇中额核心对象 从 核心对象中删除
            for instance_index in curr_C.keys():
                if instance_index in self.Ω:
                    self.Ω.pop(instance_index)


    def _compute_core_instance(self):
        """
        核心对象
        :return:
        """
        self._compute_eps_distance()
        for instance_index in  range(self.instance_num):
            if len(self.eps_neighbours[instance_index]) >= self.min_samples:
                self.Ω.update({instance_index:''})



    def _compute_eps_distance(self):
        """
        计算eps下的所有样本
        :return:
        """
        self.eps_neighbours = []
        # self.the_k_distance = np.zeros(self.instance_num)

        for instance_index in range(self.instance_num):
            eps_neighbour = []
            for neighbour_index, _instance in enumerate(self.instances):
                # print(self.distance_function(self.instances[instance_index], self.instances[neighbour_index]))
                if self.distance_function(self.instances[instance_index], self.instances[neighbour_index]) <= self.eps \
                        and instance_index != neighbour_index:
                    eps_neighbour.append(neighbour_index)
            self.eps_neighbours.append(eps_neighbour)


    def _euclidean(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)




def create_data(centers,num=100,std=0.8):
    X,labels_true = make_blobs(n_samples=num,centers=centers, cluster_std=std, random_state=1)
    return X,labels_true

def plot_data(data):
    X,labels_true = data
    labels=np.unique(labels_true)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbycm'
    for i,label in enumerate(labels):
        position = labels_true==label
        ax.scatter(X[position,0],X[position,1],label="cluster %d"%label),
        color=colors[i%len(colors)]

    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[1]")
    ax.set_title("data")
    plt.show()

if __name__=="__main__":

    X, label = create_data(4)
    # plot_data((X,label))

    # eps=1.5 时 全部纳入簇中
    dbscan = DBSCAN(X, eps=1.5, min_samples=5)
    label_pred = dbscan.predict()
    plot_data((X, label_pred))