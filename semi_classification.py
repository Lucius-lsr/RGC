# -*- coding: utf-8 -*-

from rgc import RGC
from data import get_data
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def rgc_acc(alpha, beta, mu, train_ratio, random_state, dataset):
    x, y, num_train, num_class = get_data(dataset, train_ratio, random_state)
    print('data shape: ', x.shape)
    print('number of train data: ', num_train)
    print('number of class: ', num_class)

    # load model
    model = RGC(k=5, alpha=alpha, beta=beta, mu=mu)

    # divide
    x_train = x[:num_train]
    x_test = x[num_train:]
    y_train = y[:num_train]
    y_test = y[num_train:]

    # predict
    y_pred = model.semi_classification(num_class, x_train, x_test, y_train)
    acc = np.mean(y_pred == y_test)

    return acc


for ds in ['coil20','jaffe','yale']:
    print('dataset is ', ds)
    for tr in [0.1, 0.3, 0.5]:
        print('training ratio is {}'.format(tr))
        acc_all = 0
        for rs in range(10):
            acc = rgc_acc(alpha=0.0385, beta=0.01, mu=15, train_ratio=tr, random_state=rs, dataset=ds)
            acc_all += acc
        acc_all /= 10
        print('average acc is ', acc_all)


# cls = OneVsRestClassifier(SVC(kernel='linear'))
# cls.fit(x_train, y_train)
# output = cls.predict(x_test)
# target = y_test
# acc2 = np.mean(target == output)
# print('standard ACC:', acc2)
