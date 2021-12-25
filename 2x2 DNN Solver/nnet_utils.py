'''
NEURAL NETWORK HELPER FUNCTIONS
'''

import sys
import time
import numpy as np

from cube_environment import shuffle
from cube_environment import possibleMoves



''' 
This function takes a model and returns optimized parameters. It uses the Deep
Approximate Value Iteration approach described on the paper, where it iteratively
updated the heuristic function in a way that the network minimizes the MSE between
the current estimate and a good one from a old iteration
'''
def DAVI(model, B, K, M, C, tol):
    
    theta = model.get_weights()       # get current model weights
    est_theta = theta                 # assign theta to be updated to current weights

    # for each training iteration
    for m in range(M):
        
        start = time.perf_counter()
        
        # get B shuffled states in the one hot representation
        X, Y = [shuffle(K)[0] for i in range(B)], []

        # for each training sample
        for i, x in enumerate(X):

            if i % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write(f"\rTraining iteration {m} with {i} samples labeled")

            # compute y by taking the min(1+J(nextState)) among all moves from current state x using old theta
            y = min([1 + J(model, est_theta, x.rotate(move).oneHot) for move in possibleMoves])
            Y.append(y)
        
        # train the model with new training set and labels computed using J(x)
        X, Y = np.array([x.oneHot for x in X]).reshape(B, 144), np.array(Y).reshape(-1, 1)
        theta, loss = train(model, theta, X, Y)
        
        finish = time.perf_counter()
        print(f"\rTraining iteration {m} took {round((finish - start)/60, 2)} minutes")
  
        if (M % C == 0) and (loss < tol): est_theta = theta
    
    sys.stdout.flush() 
        
    return theta




'''
This function should receive a network model and retrain it by fitting
the given data and labels. The data is the flattened state of the cube
in the one hot representation. The labels are computed the computed ys
based on J(x) and on the network with parameters theta.
'''
def train(model, theta, data, labels):
    model.set_weights(theta)
    model.fit(data, labels, epochs = 10, batch_size = 10, verbose=0)  # Training model
    theta = model.get_weights()                                       # Getting model parameters
    loss = model.evaluate(data, labels)[0]                            # Evaluating model to obtain loss
    return theta, loss                                                # Returning parameters and loss





'''
This function should return the prediction made by the network regarding
the output of J(x), where x is a given state. These predictions are used
to compute y for a state in DAVI function.
'''
def J(model, theta, state):
    model.set_weights(theta)                    # Setting theta parameters
    prediction = model.predict(state)           # Getting prediction
    return prediction                           # Returning prediction




