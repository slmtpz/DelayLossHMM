from DelayLossHMM import DelayLossHMM
from parser import data_generator

### Setting Input Parameters

# number of possible states
N = 2

# training sequence
y = data_generator()  # y1, y2, .. , yN

# stopping parameter
epsilon = 1  # TODO: set it to the right epsilon based on convergence rate for parameter likelihoods


### Initialization Step
hmm = DelayLossHMM(N, y)


### Iterative Step

new, old = 1, 1  # likelihood of parameters, we want the likelihoods for parameters to converge as much as possible
### EM algorithm
while new - old > epsilon: # check if converged
    log_alpha, log_beta, log_ro = hmm.forward_backward()
    # update A,p,d,sigma
    # likelihood = sum(alpha)
    pass

### Viterbi

# now that parameters learned, time to find best sequence of states for hidden states
seq = hmm.viterbi()

### Results, graphs, prediction(?)

