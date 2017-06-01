import matplotlib.pyplot as plt
import numpy as np
import random


def data_generator():
    t = 1
    t_p = [0.001, 0.009, 0.012]  # trend change prob, respectively prob of changing from stay, increase, decrease trends
    s = 200  # starting value
    e_p = 0.03  # loss probability
    data = np.ones(3000,dtype=int)
    for i in range(3000):
        data[i] = t
        if random.random() < t_p[t]:
            cands = [ 0, 1, 2]
            cands.remove(t)
            t = random.choice(cands)
    plt.plot(data)
    plt.show()
    return data

