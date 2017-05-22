from DelayLossHMM import DelayLossHMM
import numpy as np
from scipy.stats import gamma
from

### Setting Input Parameters

# number of possible states
N = 2

# training sequence
y = [1,1,1] # y1, y2, .. , yN

# stopping parameter
epsilon = 1

### Initialization Step

# transition probability matrix with each index being 1/N
A = np.ones([N, N]) / N

# loss probability vector notating probability of packet loss for each state
p = np.ones(N) / 2

# starting gamma parameters for gamma distribution
gammas = np.arange(1, N+1) * np.max(y) / (N + 1)

# d = y*v, sigma^2 = y*v^2, later to be used to derive d (mean delay) and sigma (variance)
v = np.ones(N)

### Iterative Step

new, old = 1, 1
while new - old > epsilon:
    pass


# state conditioned delay probability density function (pdf)
def f(i, t):
    return gamma.pdf(t / v[i], gammas[i]) / v[i]
