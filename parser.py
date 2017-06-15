import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
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

def parse(input_file):
    file = open(input_file, 'r')
    data = np.fromregex(file,'(\d+)\(\d+\%\)', [('num',np.int32)])
    res = np.empty((len(data)/16,16), dtype=np.int32)
    for i in range(len(data)/16):
        for j in range(16):
            res[i,j] = data[i*16+j][0]
    return np.argmax(np.diff(res,axis=0),axis=1)

data = parse('train1.txt')

plt.plot(data)
plt.xlabel('time')
plt.ylabel('delay level')
pylab.title('Input Delays')
plt.show()