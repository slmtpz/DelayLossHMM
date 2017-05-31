# import matplotlib.pyplot as plt
import numpy as np
import random


def data_generator():
    t = 1  # trend (-1, 0, 1)
    t_p = [0.1, 0.4, 0.4]  # trend change prob, respectively prob of changing from stay, increase, decrease trends
    s = 200  # starting value
    e_p = 0.03  # loss probability
    data = []
    for i in range(3000):
        if s <= 30:
            t = 1
        s += np.random.normal(5*t, 5)
        if random.random() < e_p:
            data.append(-1)
        else:
            data.append(s)
        if random.random() < t_p[t]:
            cands = [-1, 0, 1]
            cands.remove(t)
            t = random.choice(cands)
    # plt.plot(data)
    # plt.show()
    return data
