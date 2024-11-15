import numpy as np

class linear:
    def __init__(self, d1, d2) -> None:
        self.w1 = np.random((d1, d2))
    
    def forward(self, x):
        ...
        return x @ self.w1

    def backward(self, x, deltas, thetas):
        ...
        return deltas * x @ deltas
    
def relu(x):
    return x * (x>=0)

class MLP:
    def __init__(self, input_dim, dimensions) -> None:
        self.layers = []
        for d in dimensions:
            self.layers.append(linear(input_dim, d))
            input_dim = d
    
    def forward(self, x):
        for layer in self.layer[:-1]:
            x = relu(layer.forward(x))