# -*- coding: utf-8 -*-

from rgc import RGC
from data import get_data
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
import numpy as np
from svm import *

train_ratio = 0.5

model = RGC(10, 0.1, 1, 1)

x, y = load_iris(True)

x, y = shuffle(x, y, random_state=66)

x_size = x.shape[0]
num_train = int(x_size*train_ratio)
num_class = max(y)+1

x_train = x[:num_train]
x_test = x[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]

y_pred = model.semi_classification(num_class, x_train, x_test, y_train)
acc1 = np.mean(y_pred == y_test)

cfg = SConfig()
target, output = train_svm(**cfg.inputs, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
acc2 = np.mean(target == output)

print(acc1, acc2)
