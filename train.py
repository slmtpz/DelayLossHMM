from DelayLossHMM import DelayLossHMM
import numpy as np
from scipy.stats import gamma
from sympy.functions.special.delta_functions import DiracDelta

### Setting Input Parameters

# number of possible states
N = 2

# training sequence
y = [1,1,1] # y1, y2, .. , yN

# stopping parameter
epsilon = 1  # TODO: set it to the right epsilon based on convergence rate for parameter likelihoods

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

new, old = 1, 1  # likelihood of parameters, we want the likelihoods for parameters to converge as much as possible
while new - old > epsilon: # check if converged
    ### EM algorithm
    # a = hmm.forward()
    # b = hmm.backward()
    # ro = ...
    # update parameters
    # find new likelihood, new = sum(a)
    pass

### Viterbi

# now that parameters learned, time to find best sequence of states for hidden states
# hmm.viterbi()

### Results, graphs, prediction(?)

### Helpers
# may be moved to hmm class or some other helper py file

# state conditioned delay probability density function (pdf)
def f(i, t):
    return gamma.pdf(t / v[i], gammas[i]) / v[i]


# b is later 'corrected' to b_modified in the paper, # TODO: idk if we should use b or use only b_mod yet.
def b(i, t):
    return p[i] * f(i, t) + (1 - p[i]) * DiracDelta(t + 1)


def b_mod(i, t):
    return p[i] * f(i, t) + (1 - p[i]) * g(t)


def g(t):
    delta = 0.01 # set to be an arbitrary small number # TODO: change the value ?
    return 1 / 2 * delta if -1-delta <= t <= -1+delta else 0
