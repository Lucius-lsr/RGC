# -*- coding: utf-8 -*-

from rgc import RGC
from data import get_data
import numpy as np
from svm import *

from sklearn.tree import DecisionTreeClassifier

# load data
x, y, num_train, num_class = get_data('jaffe', 0.5)
print('data shape: ', x.shape)
print('number of train data: ', num_train)
print('number of class: ', num_class)

# load model
model = RGC(k=5, alpha=0.01, beta=0.01, mu=1)

# divide
x_train = x[:num_train]
x_test = x[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]

# predict
y_pred = model.semi_classification(num_class, x_train, x_test, y_train)
acc1 = np.mean(y_pred == y_test)

# simple compare
# cfg = SConfig()
# target, output = train_svm(**cfg.inputs, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
cls = DecisionTreeClassifier()
cls.fit(x_train, y_train)
output = cls.predict(x_test)
target = y_test
acc2 = np.mean(target == output)

print(acc1, acc2)
