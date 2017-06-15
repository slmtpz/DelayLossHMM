import numpy as np
from parser import parse
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


class DelayLossHMM(object):
    def __init__(self, N,O):
        self.S = N  # number of states
        ### Initialization Step

        self.EPSILON = 0.00001

        # transition probability matrix with each index being 1/N
        self.A = np.random.rand(N, N) + 2*np.eye(N,N)

        self.pi = np.random.rand(N)

        # emission matrix
        prior = np.eye(O,N)
        prior[:,N-1] = 0.5
        self.B = np.random.rand(O, N) + prior

        self.R = self.B.shape[0]

        self.set_logs()

    def generate_sequence(self, T=10):
        # T: Number of steps

        x = np.zeros(T, int)
        y = np.zeros(T, int)

        for t in range(T):
            if t == 0:
                x[t] = self.randgen(self.pi)
            else:
                x[t] = self.randgen(self.A[:, x[t - 1]])
            y[t] = self.randgen(self.B[:, x[t]])

        return y, x

    def forward(self, y, maxm=False):
        T = len(y)

        # Forward Pass

        # Python indices start from zero so
        # log \alpha_{k|k} will be in log_alpha[:,k-1]
        # log \alpha_{k|k-1} will be in log_alpha_pred[:,k-1]
        log_alpha = np.zeros((self.S, T))
        log_alpha_pred = np.zeros((self.S, T))
        for k in range(T):
            if k == 0:
                log_alpha_pred[:, 0] = self.logpi
            else:
                if maxm:
                    log_alpha_pred[:, k] = self.predict_maxm(log_alpha[:, k - 1])
                else:
                    log_alpha_pred[:, k] = self.predict(log_alpha[:, k - 1])

            log_alpha[:, k] = self.update(y[k], log_alpha_pred[:, k])

        return log_alpha, log_alpha_pred

    def backward(self, y, maxm=False):
        # Backward Pass
        T = len(y)
        log_beta = np.zeros((self.S, T))
        log_beta_post = np.zeros((self.S, T))

        for k in range(T - 1, -1, -1):
            if k == T - 1:
                log_beta_post[:, k] = np.zeros(self.S)
            else:
                if maxm:
                    log_beta_post[:, k] = self.postdict_maxm(log_beta[:, k + 1])
                else:
                    log_beta_post[:, k] = self.postdict(log_beta[:, k + 1])

            log_beta[:, k] = self.update(y[k], log_beta_post[:, k])

        return log_beta, log_beta_post

    def baum_welch(self, y):
        T = len(y)
        for i in range(30):
            log_alpha, log_alpha_pred = self.forward(y)
            log_beta, log_beta_post = self.backward(y)

            gamma = self.calculate_gamma(log_alpha, log_beta)
            xi = self.calculate_xi(y, log_alpha, log_beta_post)

            self.update_params(y,gamma,xi)
        self.pi = self.steady_state_probs()
        self.set_logs()

    def steady_state_probs(self):
        log_alpha, log_alpha_pred = self.forward(y)
        log_beta, log_beta_post = self.backward(y)
        return self.normalize(np.sum(self.calculate_gamma(log_alpha, log_beta), axis=1))


    def viterbi_maxsum(self, y):
        '''Vanilla implementation of Viterbi decoding via max-sum'''
        '''This algorithm may fail to find the MAP trajectory as it breaks ties arbitrarily'''
        log_alpha, log_alpha_pred = self.forward(y, maxm=True)
        log_beta, log_beta_post = self.backward(y, maxm=True)

        log_delta = log_alpha + log_beta_post
        return np.argmax(log_delta, axis=0)

    ### Helpers
    def calculate_gamma(self, log_alpha, log_beta):
        T = len(log_beta[0])
        log_gamma = log_alpha + log_beta
        gamma = np.empty((self.S, T))
        for i in range(len(log_gamma[0])):
            gamma[:, i] = self.normalize_exp(log_gamma[:, i])

        return gamma

    def calculate_xi(self, y, log_alpha, log_beta_post):
        T = len(y)
        S = self.S
        log_xi = np.empty([S, S, T])
        for i in range(S):
            for j in range(S):
                for t in range(T - 1):
                    log_xi[i, j, t] = log_alpha[i, t] + log_beta_post[j, t] + self.A[i, j] + self.B[y[t + 1], j]

        xi = np.empty([S, S, T])
        for i in range(len(log_xi[0, 0])):
            xi[:, :, i] = self.normalize_exp(log_xi[:, :, i])

        return xi

    def update_params(self, y, gamma, xi):
        T = len(y)

        self.pi = gamma[:,0]
        gamma_i = np.sum(gamma,axis=1)
        a_star = np.sum(xi,axis=(2))
        for i in range(len(a_star)):
            a_star[i, :] = np.divide(a_star[i, :], gamma_i)
        self.A = a_star
        for i in range(self.S):
            for j in range(self.R):
                sum=0
                for t in range(T):
                    if y[t] == j:
                        sum += gamma[i, t]
                self.B[j, i] = sum/gamma_i[i]

        self.set_logs()

    def randgen(self,pr, N=1):
        L = len(pr)
        return int(np.random.choice(range(L), size=N, replace=True, p=pr))

    def normalize_exp(self, log_P, axis=None):
        a = np.max(log_P, keepdims=True, axis=axis)
        P = self.normalize(np.exp(log_P - a), axis=axis)
        return P

    def normalize(self, A, axis=None):
        Z = np.sum(A, axis=axis, keepdims=True)
        idx = np.where(Z == 0)
        Z[idx] = 1
        return A / Z

    def set_logs(self):
        for i in range(self.S):
            self.B[:, i] = self.normalize(self.B[:, i]+self.EPSILON)
        for i in range(self.S):
            self.A[:, i] = self.normalize(self.A[:, i]+self.EPSILON)
        self.pi = self.normalize(self.pi+self.EPSILON)
        self.logB = np.log(self.B)
        self.logA = np.log(self.A)
        self.logpi = np.log(self.pi)

    def predict(self, lp):
        lstar = np.max(lp)
        return lstar + np.log(np.dot(self.A, np.exp(lp - lstar)))

    def postdict(self, lp):
        lstar = np.max(lp)
        return lstar + np.log(np.dot(np.exp(lp - lstar), self.A))

    def predict_maxm(self, lp):
        return np.max(self.logA + lp, axis=1)

    def postdict_maxm(self, lp):
        return np.max(self.logA.T + lp, axis=1)

    def update(self, y_k, lp):
        return self.logB[y_k, :] + lp if not np.isnan(y_k) else lp


hmm4 = DelayLossHMM(4,16)
hmm3 = DelayLossHMM(3,16)
hmm2 = DelayLossHMM(2,16)

y = parse('train1.txt')
hmm2.baum_welch(y)
hmm3.baum_welch(y)
hmm4.baum_welch(y)

seq, x = hmm3.generate_sequence(300)



plt.plot(seq)
plt.xlabel('time')
plt.ylabel('delay level')
pylab.title('Prediction')
plt.show()

seq = hmm2.viterbi_maxsum(y)
plt.plot(seq)
plt.xlabel('time')
plt.ylabel('State Number')
pylab.title('Output of the Viterbi Algorithm for 2 states')
plt.show()

seq = hmm3.viterbi_maxsum(y)
plt.plot(seq)
plt.xlabel('time')
plt.ylabel('State Number')
pylab.title('Output of the Viterbi Algorithm for 3 states')
plt.show()

seq = hmm4.viterbi_maxsum(y)
plt.plot(seq)
plt.xlabel('time')
plt.ylabel('State Number')
pylab.title('Output of the Viterbi Algorithm for 4 states')
plt.show()
