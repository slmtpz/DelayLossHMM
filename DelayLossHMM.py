import numpy as np
from scipy.stats import gamma
from sympy.functions.special.delta_functions import DiracDelta


class DelayLossHMM(object):
    def __init__(self, N, y):
        self.N = N  # number of states
        self.y = y  # training sequence TODO: may be moved to a train function
        ### Initialization Step

        # transition probability matrix with each index being 1/N
        self.A = np.ones([N, N]) / N

        # loss probability vector notating probability of packet loss for each state
        self.p = np.ones(N) / 2

        # starting gamma parameters for gamma distribution
        self.gammas = np.arange(1, N + 1) * np.max(y) / (N + 1)

        # d = y*v, sigma^2 = y*v^2, later to be used to derive d (mean delay) and sigma (variance)
        self.v = np.ones(N)

    def forward_backward(self):
        log_alpha = [1, 1, 1]
        log_beta = [1, 1, 1]
        log_ro = [1, 1, 1]
        return log_alpha, log_beta, log_ro

    def viterbi(self):
        seq = [1, 1, 1]
        return seq


    ### Helpers
    # may be moved to hmm class or some other helper py file

    # state conditioned delay probability density function (pdf)
    def f(self, i, t):
        return gamma.pdf(t / self.v[i], self.gammas[i]) / self.v[i]


    # b is later 'corrected' to b_modified in the paper, # TODO: idk if we should use b or use only b_mod yet.
    def b(self, i, t):
        return self.p[i] * self.f(i, t) + (1 - self.p[i]) * DiracDelta(t + 1)


    def b_mod(self, i, t):
        return self.p[i] * self.f(i, t) + (1 - self.p[i]) * g(t)


    def g(self, t):
        delta = 0.01 # set to be an arbitrary small number # TODO: change the value ?
        return 1 / 2 * delta if -1-delta <= t <= -1+delta else 0
