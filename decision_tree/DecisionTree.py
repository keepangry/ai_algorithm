# encoding: utf-8
'''
@author: yangsen
@license:
@contact: 0@keepangry.com
@software:
@file: DecisionTree.py
@time: 18-8-23 下午9:14
@desc:
 https://blog.csdn.net/ruggier/article/details/78756447#id3%E7%AE%97%E6%B3%95
实现 ID3  和  C4.5

'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from util.entropy import discrete_entropy, gini
import numpy as np
from collections import Counter
from sklearn import datasets, neighbors, linear_model
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as Tree
from sklearn.model_selection import train_test_split

class DecisionTree(object):
    train_data = np.zeros((2,2))
    train_label = np.zeros(2)
    train_num = 2
    feature_num = 2
    threshold = 0
    feature_values = []  # 索引为特征序号，值所有可能取值的列表
    method='id3'  # id3 / c4.5

    tree = {}   # {0: {0: 0, 1: 1, 'feature': 1}, 1: 1, 'feature': 2}  用字典表示。  特征值是离散的，如果是叶子，则是label，如果非叶子，则递归表示树。


    def __init__(self, threshold=0):
        self.threshold = threshold

    def fit(self, train_data, train_label, method='id3', is_continuous=False, info_gain='entropy'):
        self.train_data = train_data
        self.train_label = train_label
        self.labels = Counter(train_label).values()
        self.train_num = self.train_data.shape[0]
        self.feature_num = self.train_data.shape[1]
        ## 用来获取索引
        self.train_indexes = np.arange(self.train_num)
        self.feature_indexes = np.arange(self.feature_num)
        self.method = method
        self.is_continuous = is_continuous
        self.info_gain = info_gain
        if self.info_gain == 'entropy':
            self.entropy = discrete_entropy
        elif self.info_gain == 'gini':
            self.entropy = gini


        ## 初始化，首先遍历每个特征的所有取值可能性
        self.feature_values = [list(set(self.train_data[:, feature_index])) for feature_index in
                               range(self.feature_num)]
        if self.is_continuous:
            self.tree = self.__create_branch_continuous(train_indexes=np.arange(self.train_num),
                                                     feature_indexes=np.arange(self.feature_num))
        else:
            self.tree = self._create_branch_discrete(train_indexes=np.arange(self.train_num),
                                                 feature_indexes=np.arange(self.feature_num))

    def _info_gain(self, train_indexes, feature_index):
        """
        信息增益，导致熵减。减小的值是便是信息增益，越大越好。

        :param train_indexes:
        :param feature_index:
        :param is_ration: 是否使用信息增益率
        :return:
        """

        # 所有可能的取值
        feature_values = self.feature_values[feature_index]

        # 混乱，熵大
        origin_entropy = self.entropy(self.train_label[train_indexes])

        new_entropy = 0
        # 计算所有可能取值下，剩余的熵
        for feature_value in feature_values:
            label_vector = self.train_label[train_indexes] [ self.train_data[train_indexes][:,feature_index]==feature_value ]

            if label_vector.shape[0] != 0:
                new_entropy += label_vector.shape[0]/train_indexes.shape[0] * self.entropy(label_vector)

        if self.method!='id3':
            return (origin_entropy - new_entropy + 0.1) / (new_entropy + 0.1)
        else:
            return origin_entropy - new_entropy

    def _info_gain_split(self, train_indexes, feature_index, split_value):
        """

        :param train_indexes:
        :param feature_index:
        :param split_value:
        :return:
        """
        # 混乱，熵大
        origin_entropy = self.entropy(self.train_label[train_indexes])

        new_entropy = 0
        # 计算所有可能取值下，剩余的熵

        label_vector_left = self.train_label[train_indexes][
            self.train_data[train_indexes][:, feature_index] < split_value]
        label_vector_right = self.train_label[train_indexes][
            self.train_data[train_indexes][:, feature_index] >= split_value]
        new_entropy = label_vector_left.shape[0] / train_indexes.shape[0] * self.entropy(label_vector_left) + \
                      label_vector_right.shape[0] / train_indexes.shape[0] * self.entropy(label_vector_right)
        if self.method != 'id3':
            return (origin_entropy - new_entropy + 0.1) / (new_entropy + 0.1)
        else:
            return origin_entropy - new_entropy



    def __feature_gain(self, train_indexes, feature_index):
        """
        计算特征的最佳收益分裂点

        :return:
        """
        sorted_value = sorted(list(set(self.train_data[train_indexes][:, feature_index])))
        value_num = len(sorted_value)

        # 不可能为0，因为train_indexes时需要检查

        # 该特征完全相同，但是lavel完全相同不会进来进行计算。所以，一定不会选这个特征，该特征引入不会增加信息。
        # 如果全部的特征均相同，则不在进行分裂。
        if value_num == 1:
            return 0, 0

        split_value = 0
        max_info_gain = 0
        best_split_value = 0
        for end in range(1,value_num):
            start = end-1
            split_value = (sorted_value[end] + sorted_value[start]) / 2
            # 计算当前分割点的信息增益
            info_gain = self._info_gain_split(train_indexes=train_indexes, feature_index=feature_index, split_value=split_value)
            if max_info_gain < info_gain:
                max_info_gain = info_gain
                best_split_value = split_value
        return max_info_gain, best_split_value


    def __create_branch_continuous(self, train_indexes, feature_indexes):
        """
        二分法进行。穷举所有可能的二分点。计算出的信息增益为该特征的信息增益。
        对于树结构，进行相关信息的存储以及调整。

        :param train_indexes:
        :param feature_indexes:
        :return:
        """
        label = self.train_label[train_indexes]
        # 当该分支下面所有的label都相同的时候，则直接返回label值
        label_counter = Counter(label)
        if len(label_counter)==1:
            return label[0]

        # 遍历feature，找出增益最大的那个
        info_gains = np.zeros(feature_indexes.shape[0])
        info_split_values = np.zeros(feature_indexes.shape[0])
        for index, feature_index in enumerate(feature_indexes):
            info_gains[index], info_split_values[index] = self.__feature_gain(train_indexes, feature_index)

        # 选中的进行分裂的特征
        best_feature_idx = info_gains.argmax()
        if info_gains.max() == 0:  # 如果均无信息增益，选择label最多数量最多的作为叶子label
            # recommanded_label
            max_label_num = max(label_counter.values())
            recommanded_label = dict(zip(label_counter.values(), label_counter.keys()))[max_label_num]
            return recommanded_label

        choice_feature_index = feature_indexes[best_feature_idx]
        choice_feature_split_value = info_split_values[best_feature_idx]


        # 保存
        branch = {}
        branch['feature'] = choice_feature_index
        branch['split_value'] = choice_feature_split_value

        left_branch_train_indexes = train_indexes[self.train_data[train_indexes][:, choice_feature_index] < choice_feature_split_value]
        right_branch_train_indexes = train_indexes[self.train_data[train_indexes][:, choice_feature_index] >= choice_feature_split_value]
        branch['left'] = self.__create_branch_continuous(left_branch_train_indexes, feature_indexes)
        branch['right'] = self.__create_branch_continuous(right_branch_train_indexes, feature_indexes)

        return branch


    def _create_branch_discrete(self, train_indexes, feature_indexes):
        """
        数据为离散变量。分支使用离散变量的值进行决策

        创建新的树枝，肯定要在当前剩下的样本上进行创建。 比如，选中性别的，需要筛选出性别符合的样本往下进行。
        当该特征用过之后，当前也不存在该特征，需要提出该特征。
        :param train_indexes:
        :param feature_indexes:
        :return:
        """
        """

        :return:
        """
        label = self.train_label[train_indexes]

        # 当该分支下面所有的label都相同的时候，则直接返回label值
        label_counter = Counter(label)
        if len(label_counter)==1:
            return label[0]

        # recommanded_label
        max_label_num = max(label_counter.values())
        recommanded_label = dict(zip(label_counter.values(), label_counter.keys()))[max_label_num]

        # 遍历feature，找出增益最大的那个
        info_gains = np.zeros(feature_indexes.shape[0])
        for index, feature_index in enumerate(feature_indexes):
            info_gains[index] = self._info_gain(train_indexes, feature_index)

        # 选中的进行分裂的特征
        choice_feature_index = feature_indexes[info_gains.argmax()]

        # 保存
        branch = {}
        branch['feature'] = choice_feature_index

        for feature_value  in self.feature_values[choice_feature_index]:
            # 如果该分支下该value无样本，则推荐该分之下最多样本那一类
            branch_train_indexes = train_indexes [ self.train_data[train_indexes][:, choice_feature_index] == feature_value ]
            if branch_train_indexes.shape[0] == 0:
                branch[feature_value] = recommanded_label
            else:
                branch[feature_value] = self._create_branch_discrete(
                    train_indexes=self.train_indexes[train_indexes] [ self.train_data[train_indexes][:, choice_feature_index] == feature_value ],
                    feature_indexes=feature_indexes[ feature_indexes != choice_feature_index ]
                )
        return branch



    def predict(self, sample2D):
        # 递归寻找叶子节点
        def pred_sample(tree, sample1D):
            if type(tree) != dict:
                return tree
            else:
                if sample1D[tree['feature']] not in tree:
                    print ("出现未登录特征值，进行随机判断") # 第一个判断
                    return self.labels[0]
                else:
                    return pred_sample(tree[sample1D[tree['feature']]], sample1D)

        # 连续变量特征
        def pred_sample__continuous(tree, sample1D):
            if type(tree) != dict:
                return tree
            else:
                if sample1D[tree['feature']] < tree['split_value']:
                    return pred_sample__continuous(tree['left'], sample1D)
                else:
                    return pred_sample__continuous(tree['right'], sample1D)

        result = np.zeros(sample2D.shape[0])
        for index, sample1D in enumerate(sample2D):
            if self.is_continuous:
                result[index] = pred_sample__continuous(self.tree, sample1D)
            else:
                result[index] = pred_sample(self.tree, sample1D)
        return result


if __name__=="__main__":
    train_data = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 0, 1, 2],
        [1, 0, 1, 2],
        [2, 0, 1, 2],
        [2, 0, 1, 1],
        [2, 1, 0, 1],
        [2, 1, 0, 2],
        [2, 0, 0, 0],
    ])

    train_label = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    tree = DecisionTree(threshold=0)
    tree.fit(train_data=train_data, train_label=train_label)
    print(tree.predict( np.array([[1,0,1,0], [0,1,0,1]]) ))

    # 准确率测试实验


    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    # 降到二分类
    X_digits = np.vstack((X_digits[ y_digits==2 ],X_digits[ y_digits==1 ]))
    y_digits = np.append(y_digits[ y_digits==2 ],y_digits[ y_digits==1 ])

    n_samples = len(X_digits)

    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size = 0.2, random_state = 42)

    knn = neighbors.KNeighborsClassifier()
    logistic = linear_model.LogisticRegression()
    tree1 = DecisionTree()
    tree2 = DecisionTree()

    DecisionTreeClassifier = DecisionTreeClassifier()

    tree1.fit(train_data=X_train, train_label=y_train)
    print('my decisionTree 1 score: %f'
          % accuracy_score(tree1.predict(X_test), y_test))

    DecisionTreeClassifier.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    logistic.fit(X_train, y_train)

    print('DecisionTree score: %f' % accuracy_score(DecisionTreeClassifier.predict(X_test), y_test) )
    print('KNN score: %f' % accuracy_score(knn.predict(X_test), y_test) )
    print('LogisticRegression score: %f'
          % accuracy_score(logistic.predict(X_test), y_test))

    tree2.fit(train_data=X_train, train_label=y_train, method='c4.5')
    print('my decisionTree 2 score: %f'
          % accuracy_score(tree2.predict(X_test), y_test))
    tree3 = DecisionTree()
    tree3.fit(train_data=X_train, train_label=y_train, is_continuous=True, method='c4.5')
    print('my decisionTree 3 score: %f'
          % accuracy_score(tree3.predict(X_test), y_test))
    tree4 = DecisionTree()
    tree4.fit(train_data=X_train, train_label=y_train, is_continuous=True, method='c4.5', info_gain='gini')
    print('my decisionTree 4 gini score: %f'
          % accuracy_score(tree4.predict(X_test), y_test))
