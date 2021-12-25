'''
THE DEEP NEURAL NETWORK MODEL
'''

from keras.models import Model
from keras.layers import Input, Dense, Add, BatchNormalization


class Network:
    
    def __init__(self, nout = 1, loss='mean_squared_error', metrics='mean_squared_error'):
        
        self.inputs = Input(shape=(144))
        
        # first two hidden layers
        self.layer1 = Dense(5000, activation='relu')(self.inputs)
        self.bNorm1 = BatchNormalization()(self.layer1)
        self.layer2 = Dense(1000, activation='relu')(self.bNorm1)
        self.bNorm2 = BatchNormalization()(self.layer2)
        
        # residual layers
        self.residual1 = self.generate_residual_layer(self.bNorm2)
        self.residual2 = self.generate_residual_layer(self.residual1)
        self.residual3 = self.generate_residual_layer(self.residual2)
        self.residual4 = self.generate_residual_layer(self.residual3)
        
        # output layer
        self.outputs = Dense(nout, activation = 'relu')(self.residual4)
        
        # compile model
        self.model = Model(inputs = self.inputs, outputs = self.outputs)
        self.model.compile(optimizer='adam', loss=loss, 
                           metrics=[metrics])



    
    # generates a residual layer composed of two dense layers and batch norm layers
    def generate_residual_layer(self, previous_layer):
    
        x1 = Dense(1000, activation='relu')(previous_layer)
        b1 = BatchNormalization()(x1)
    
        x2 = Dense(1000, activation='relu')(b1)
        b2 = BatchNormalization()(x2)
    
        residual = Add()([x1, b1, x2, b2])
        residual = BatchNormalization()(residual)
        
        return residual


    
