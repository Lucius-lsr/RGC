# -*- coding: utf-8 -*-

from rgc import RGC
from data import get_separated_data
import numpy as np


def rgc_acc(alpha, beta, mu, train_ratio, random_state, dataset):
    x_train, x_test, y_train, y_test, num_class = get_separated_data(dataset, train_ratio, random_state)

    # load model
    model = RGC(k=5, alpha=alpha, beta=beta, mu=mu)

    # predict
    y_pred = model.semi_classification(num_class, x_train, x_test, y_train)
    acc = np.mean(y_pred == y_test)
    return acc


dataset = 'jaffe'
train_ratio = 0.5
acc = rgc_acc(alpha=0.0385, beta=0.1, mu=15, train_ratio=train_ratio, random_state=66, dataset=dataset)
print('accuracy is ', acc)

