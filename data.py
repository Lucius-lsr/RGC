# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import scipy.io
import random


def get_data(dataset_name, train_ratio, random_state):
    # train_ratio = 0.5
    # x, y = load_iris(True)
    # x, y = shuffle(x, y, random_state=66)
    # x_size = x.shape[0]
    # num_train = int(x_size * train_ratio)
    # num_class = max(y) + 1
    # return x, y, num_train, num_class

    dataset_dict = {'coil20': 'dataset/COIL20_1440n_1024d_20c.mat',
                    'jaffe': 'dataset/jaffe_213n_676d_10c_uni.mat',
                    'yale': 'dataset/YALE_165n_1024d_15c_uni.mat'}

    try:
        data_path = dataset_dict[dataset_name]
        mat = scipy.io.loadmat(data_path)
        x = np.array(mat['X'])
        y = np.array(mat['y']).reshape(-1)
        y -= 1
        x, y = shuffle(x, y, random_state=random_state)
        x_size = x.shape[0]
        num_train = int(x_size * train_ratio)
        num_class = np.max(y) - np.min(y) + 1
        return x, y, num_train, num_class
    except Exception as e:
        print(e)
        exit(1)


def get_separated_data(dataset_name, train_ratio, random_state):
    dataset_dict = {'coil20': 'dataset/COIL20_1440n_1024d_20c.mat',
                    'jaffe': 'dataset/jaffe_213n_676d_10c_uni.mat',
                    'yale': 'dataset/YALE_165n_1024d_15c_uni.mat'}

    try:
        data_path = dataset_dict[dataset_name]
        mat = scipy.io.loadmat(data_path)
        x = np.array(mat['X'])
        y = np.array(mat['y']).reshape(-1)
        y -= 1
        num_class = np.max(y) - np.min(y) + 1

        x_size = x.shape[0]
        each_class_begin = [0]
        for i in range(x_size - 1):
            assert y[i] <= y[i + 1]
            if y[i] < y[i + 1]:
                each_class_begin.append(i + 1)
        each_class_begin.append(x_size - 1)

        train_idx = []
        train_num_per_class = int(x_size * train_ratio / num_class)

        random.seed(random_state)

        for i in range(num_class):
            train_idx += random.sample(list(range(each_class_begin[i], each_class_begin[i + 1])), train_num_per_class)

        train_idx += random.sample(list(range(x_size)), int(x_size * train_ratio) - len(train_idx))
        train_idx = list(set(train_idx))

        test_idx = [i for i in range(x_size) if i not in train_idx]

        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        return x_train, x_test, y_train, y_test, num_class
    except Exception as e:
        print(e)
        exit(1)
