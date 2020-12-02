# -*- coding: utf-8 -*-

from rgc import RGC
from data import get_data

X = get_data()

model = RGC(10, 1, 10, 1)

model.graph_construct(X)

