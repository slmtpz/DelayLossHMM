import matplotlib.pyplot as plt
import numpy as np
import random


def data_generator():
    t = 1
    c = 0
    t_p = [0.99, 0.001, 0.99]  # trend change prob, respectively prob of changing from stay, increase, decrease trends
    s = 200  # starting value
    data = np.ones(3000,dtype=int)
    for i in range(3000):
        c += t
        c = min(max(c, 0), 9)
        data[i] = c
        if random.random() < t_p[t]:
            cands = [-1, 0, 1]
            cands.remove(t)
            t = random.choice(cands)
    plt.plot(data)
    plt.show()
    return data

