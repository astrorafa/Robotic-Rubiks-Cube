'''
CUBE AGENT
'''


import copy
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class CubeDNN:

    __slots__ = ['state', 'gval', 'parent', 'parentMove', 'fval', 'oneHot']

    # Front = r, Right = b, Back = o, Left = g, Up = w, Down = y
    GOAL = np.array(["r", "r", "r", "r", "b", "b", "b", "b", "o", "o", "o", "o",
                     "g", "g", "g", "g", "w", "w", "w", "w", "y", "y", "y", "y"])

    def __init__(self, state, parent = None, move = None, gval = 0):
        self.state      = state
        self.parent     = parent
        self.parentMove = move
        self.gval       = gval
        self.fval       = gval + 0
        self.oneHot     = self.getOneHot()


    def __eq__(self, other):
        return self.state == other.state


    def __str__(self):
        return str(str(self.state[0]))


    def __repr__(self):
        return str(str(self.state[0]))


    def __lt__(self, other):
        return self.fval < other.fval


    
    '''
    This function tests whether the current state is the goal state.
    '''
    def goalTest(self):
        if self.state == CubeDNN.GOAL: return True
        else: return False


        
    '''
    This function renders the state of the cube to allow us to visualize it.
    '''
    def render(self):
        print(" ", " ", self.state[19], self.state[16])
        print(" ", " ", self.state[18], self.state[17])
        
        print(self.state[15], self.state[12], self.state[3], self.state[0], self.state[7], self.state[4], self.state[11], self.state[8])
        print(self.state[14], self.state[13], self.state[2], self.state[1], self.state[6], self.state[5], self.state[10], self.state[9])
        
        print(" ", " ", self.state[23], self.state[20])
        print(" ", " ", self.state[22], self.state[21])

    
    
    '''
    This function performs face rotation. It returns a new cube instance, which
    has an updated state, and has its parent as the one that originated it.
    '''
    def rotate(self, move):
        if "U" in move:
            if "'" in move: new_state = self.rotateUp(-90)
            else: new_state = self.rotateUp(90)
        elif "D" in move:
            if "'" in move: new_state = self.rotateDown(-90)
            else: new_state = self.rotateDown(90)
        elif "F" in move:
            if "'" in move: new_state = self.rotateFront(-90)
            else: new_state = self.rotateFront(90)
        elif "B" in move:
            if "'" in move: new_state = self.rotateBack(-90)
            else: new_state = self.rotateBack(90)
        elif "R" in move:
            if "'" in move: new_state = self.rotateRight(-90)
            else: new_state = self.rotateRight(90)
        elif "L" in move:
            if "'" in move: new_state = self.rotateLeft(-90)
            else: new_state = self.rotateLeft(90)
        return CubeDNN(new_state, parent=self.state, move=move, gval=self.gval+1)


    
    
    def rotateFront(self, direction):

        state  = self.state
        nstate = copy.deepcopy(self.state)

        if direction == 90:
            for i in range(0, 4): nstate[i] = state[(i-1)%4]         # updates the entire front face
            nstate[6],  nstate[7]  = state[17], state[18]            # updates right face with facelets from up face
            nstate[17], nstate[18] = state[12], state[13]            # updates up face with facelets from left face
            nstate[12], nstate[13] = state[23], state[20]            # updates left face with facelets from down face
            nstate[23], nstate[20] = state[6],  state[7]             # updates down face with facelets from right face

        elif direction == -90:
            for i in range(0, 4): nstate[i] = state[(i+1)%4]         # updates the entire front face
            nstate[6],  nstate[7]  = state[23], state[20]            # updates right face with facelets from down face
            nstate[17], nstate[18] = state[6],  state[7]             # updates up face with facelets from right face
            nstate[12], nstate[13] = state[17], state[18]            # updates left face with facelets from up face
            nstate[23], nstate[20] = state[12], state[13]            # updates down face with facelets from left face

        return nstate



    
    def rotateBack(self, direction):

        state  = self.state
        nstate = copy.deepcopy(self.state)

        if direction == 90:
            for i in range(8, 12): nstate[i] = state[(i-1-8)%4+8]       # updates the entire back face
            nstate[16], nstate[19] = state[5],  state[4]                # updates up face with facelets from right face
            nstate[22], nstate[21] = state[15], state[14]               # updates down face with facelets from left face
            nstate[5],  nstate[4]  = state[22], state[21]               # updates right face with facelets from down face
            nstate[15], nstate[14] = state[16], state[19]               # updates left face with facelets from up face

        elif direction == -90:
            for i in range(8, 12): nstate[i] = state[(i+1-8)%4+8]       # updates the entire back face
            nstate[16], nstate[19] = state[15], state[14]               # updates up face with facelets from right face
            nstate[22], nstate[21] = state[5],  state[4]                # updates down face with facelets from XX face
            nstate[5],  nstate[4]  = state[16], state[19]               # updates right face with facelets from XX face
            nstate[15], nstate[14] = state[22], state[21]               # updates left face with facelets from XX face

        return nstate




    def rotateRight(self, direction):

        state  = self.state
        nstate = copy.deepcopy(self.state)

        if direction == 90:
            for i in range(4, 8): nstate[i] = state[(i-1-4)%4+4]         # updates the entire right face
            nstate[16], nstate[17] = state[0],  state[1]                 # updates up face with facelets from front face
            nstate[20], nstate[21] = state[10], state[11]                # updates down face with facelets from back face
            nstate[0],  nstate[1]  = state[20], state[21]                # updates front face with facelets from down face
            nstate[10], nstate[11] = state[16], state[17]                # updates back face with facelets from up face

        elif direction == -90:
            for i in range(4, 8): nstate[i] = state[(i+1-4)%4+4]         # updates the entire right face
            nstate[16], nstate[17] = state[10], state[11]                # updates up face with facelets from back face
            nstate[20], nstate[21] = state[0],  state[1]                 # updates down face with facelets from front face
            nstate[0],  nstate[1]  = state[16], state[17]                # updates front face with facelets from up face
            nstate[10], nstate[11] = state[20], state[21]                # updates back face with facelets from down face

        return nstate




    def rotateLeft(self, direction):

        state  = self.state
        nstate = copy.deepcopy(self.state)

        if direction == 90:
            for i in range(12, 16): nstate[i] = state[(i-1-12)%4+12]     # updates the entire left face
            nstate[18], nstate[19] = state[8],  state[9]                 # updates up face with facelets from back face
            nstate[2],  nstate[3]  = state[18], state[19]                # updates front face with facelets from up face
            nstate[22], nstate[23] = state[2],  state[3]                 # updates down face with facelets from front face
            nstate[8],  nstate[9]  = state[22], state[23]                # updates back face with facelets from down face

        elif direction == -90:
            for i in range(12, 16): nstate[i] = state[(i+1-12)%4+12]     # updates the entire left face
            nstate[18], nstate[19] = state[2],  state[3]                 # updates up face with facelets from front face
            nstate[2],  nstate[3]  = state[22], state[23]                # updates front face with facelets from down face
            nstate[22], nstate[23] = state[8],  state[9]                 # updates down face with facelets from back face
            nstate[8],  nstate[9]  = state[18], state[19]                # updates back face with facelets from up face

        return nstate




    def rotateUp(self, direction):

        state  = self.state
        nstate = copy.deepcopy(self.state)

        if direction == 90:
            for i in range(16, 20): nstate[i] = state[(i-1-16)%4+16]        # updates the entire up face
            nstate[0],  nstate[3]  = state[4],  state[7]                    # updates front face with facelets from right face
            nstate[4],  nstate[7]  = state[8],  state[11]                   # updates right face with facelets from back face
            nstate[8],  nstate[11] = state[12], state[15]                   # updates back face with facelets from left face
            nstate[12], nstate[15] = state[0],  state[3]                    # updates left face with facelets from front face

        elif direction == -90:
            for i in range(16, 20): nstate[i] = state[(i+1-16)%4+16]        # updates the entire up face
            nstate[0],  nstate[3]  = state[12], state[15]                   # updates front face with facelets from left face
            nstate[4],  nstate[7]  = state[0],  state[3]                    # updates right face with facelets from front face
            nstate[8],  nstate[11] = state[4],  state[7]                    # updates back face with facelets from right face
            nstate[12], nstate[15] = state[8],  state[11]                   # updates left face with facelets from back face

        return nstate





    def rotateDown(self, direction):

        state  = self.state
        nstate = copy.deepcopy(self.state)

        if direction == 90:
            for i in range(20, 24): nstate[i] = state[(i-1-20)%4+20]        # updates the entire down face
            nstate[1],  nstate[2]  = state[13], state[14]                   # updates front face with facelets from left face
            nstate[5],  nstate[6]  = state[1],  state[2]                    # updates right face with facelets from front face
            nstate[9],  nstate[10] = state[5],  state[6]                    # updates back face with facelets from right face
            nstate[13], nstate[14] = state[9],  state[10]                   # updates left face with facelets from back face

        elif direction == -90:
            for i in range(20, 24): nstate[i] = state[(i+1-20)%4+20]        # updates the entire down face
            nstate[1],  nstate[2]  = state[5],  state[6]                    # updates front face with facelets from rigth face
            nstate[5],  nstate[6]  = state[9],  state[10]                   # updates right face with facelets from back face
            nstate[9],  nstate[10] = state[13], state[14]                   # updates back face with facelets from left face
            nstate[13], nstate[14] = state[1],  state[2]                    # updates left face with facelets from front face

        return nstate


    '''
    This function gets the one hot encoding of the categories, which means 
    that it creates a matrix where the rows represent the facelet position of 
    the cube and the columns represent the color that the given facelet position 
    has. It goes one state further and flattens the matrix so it can be given as
    an input to the DNN.
    '''
    def getOneHot(self):
        encoder = OneHotEncoder(handle_unknown='ignore')
        one_hot = encoder.fit_transform(self.state.reshape(-1, 1)).toarray()
        flattened_one_hot = one_hot.flatten().reshape(1, 144)
        return flattened_one_hot
    
    


    
