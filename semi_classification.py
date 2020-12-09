# -*- coding: utf-8 -*-

from rgc import RGC
from data import get_data
import numpy as np
from svm import *

# load data
x, y, num_train, num_class = get_data()

# load model
model = RGC(10, 0.1, 1, 1)

# divide
x_train = x[:num_train]
x_test = x[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]

# predict
y_pred = model.semi_classification(num_class, x_train, x_test, y_train)
acc1 = np.mean(y_pred == y_test)

# simple compare
cfg = SConfig()
target, output = train_svm(**cfg.inputs, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
acc2 = np.mean(target == output)

print(acc1, acc2)
