#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt
import numpy as np

N = 1000
M = 100
ensemble = np.random.rand(N, 2)
loc = (0.5, 0.5)

d = np.sum((ensemble - loc) ** 2.0, axis=-1) ** 0.5

_as = np.argsort(d)
gp = _as[:M]
rp = _as[M:]

_S = 16
for p in gp:
    x, y = ensemble[p, :].tolist()
    plt.scatter([x], [y], s=_S, color="green")

for p in rp:
    x, y = ensemble[p, :].tolist()
    plt.scatter([x], [y], s=_S, color="red")

plt.show()
