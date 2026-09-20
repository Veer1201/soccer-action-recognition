import numpy as np
from act1.engine import Value

class Neuron:

    def __init__(self, n_inputs):
        self.weights = []
        
        for i in range(n_inputs):
            weight = Value(np.random.uniform(-1, 1))
            self.weights.append(weight)
        
        self.bias = Value(np.random.uniform(-1, 1))

    def __call__(self, inputs):
        total =  Value(0.0)
        for wi, xi in zip(self.weights, inputs):
            total = total + (wi * xi)

        out = total + self.bias
        return out.tanh()
        

        
