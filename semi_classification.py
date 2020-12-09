# -*- coding: utf-8 -*-

from rgc import RGC
from data import get_data
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

x, y, num_train, num_class = get_data('jaffe', 0.5)
print('data shape: ', x.shape)
print('number of train data: ', num_train)
print('number of class: ', num_class)

# divide
x_train = x[:num_train]
x_test = x[num_train:]
y_train = y[:num_train]
y_test = y[num_train:]


def rgc_acc(alpha, beta, mu):
    # load model
    model = RGC(k=5, alpha=alpha, beta=beta, mu=mu)

    # divide
    x_train = x[:num_train]
    x_test = x[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    # predict
    y_pred = model.semi_classification(num_class, x_train, x_test, y_train, y_test)
    acc1 = np.mean(y_pred == y_test)

    print(alpha, beta, mu)
    print('RGC ACC: ', acc1)


rgc_acc(0.0385, 0.01, 15)

cls = OneVsRestClassifier(SVC(kernel='linear'))
cls.fit(x_train, y_train)
output = cls.predict(x_test)
target = y_test
acc2 = np.mean(target == output)

print('standard ACC:', acc2)
