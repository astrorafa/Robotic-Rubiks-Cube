'''
CUBE ENVIRONMENT
'''

import numpy as np
from cube_agent import CubeDNN



# List of possible moves
possibleMoves = ["U", "U'", "D", "D'", "F", "F'", "B", "B'", "R", "R'", "L", "L'"]

# Goal state
goal_state = np.array(["r", "r", "r", "r", "b", "b", "b", "b", "o", "o", "o", "o", 
                       "g", "g", "g", "g", "w", "w", "w", "w", "y", "y", "y", "y"])

# Goal node
goal = CubeDNN(goal_state)



'''
Given a K and the goal node, this function shuffles the cube to generate sample
states. It randomly select a k between 1 and K, and then samples k moves from the
list of possible moves. This function returns the final, shuffled node as CubeDNN
instance, and the number of moves k used to generate that configuration.
'''
def shuffle(K):
    node = goal
    moves = np.random.choice(possibleMoves, np.random.randint(1, K+1, 1))
    for move in moves: node = node.rotate(move)
    node.gval = 0
    node.parent = None
    node.parentMove = None
    return node, len(moves)