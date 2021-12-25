'''
MAIN FILE - computes the cost function and plots its performance against trye labels
'''

import time
import numpy as np
import matplotlib.pyplot as plt

from nnet_model import Network
from nnet_utils import J, DAVI
from cube_environment import shuffle


# training parameters
B = 10000                   # batch size
K = 30                      # max number of shuffles
M = 10                      # number of training iterations
C = 2                       # frequency of convergency check
tol = 0.05                  # loss threshold


# initilize model
network = Network()
model = network.model


# training the network
start = time.perf_counter()
trained_theta = DAVI(model, B, K, M, C, tol)
finish = time.perf_counter()
print(f'Training took {round((finish - start)/60, 2)} minutes\n')


# generating and predicting test data
test_data = np.array([shuffle(K) for _ in range(B)])
predictions = np.array([J(model, trained_theta, state[0].oneHot)[0][0] for state in test_data])


# plotting predictions and true labels
plt.scatter(test_data[:,1], predictions)
plt.show()