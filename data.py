# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_iris
from sklearn.utils import shuffle


def get_data():
    train_ratio = 0.5
    x, y = load_iris(True)
    x, y = shuffle(x, y, random_state=66)
    x_size = x.shape[0]
    num_train = int(x_size * train_ratio)
    num_class = max(y) + 1
    return x, y, num_train, num_class

