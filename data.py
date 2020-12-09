# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import scipy.io


def get_data(dataset_name, train_ratio):
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
        # x, y = shuffle(x, y, random_state=67)
        x_size = x.shape[0]
        num_train = int(x_size * train_ratio)
        num_class = np.max(y) - np.min(y) + 1
        return x, y, num_train, num_class
    except Exception as e:
        print(e)
        exit(1)
