import numpy as np

class Linear:
    def __init__(self, d1, d2) -> None:
        # d1: number of nodes in the previous layer  d2: number of nodes in the current layer
        self.w = np.random.randn(d1, d2) * 0.01     # random weight normal distribution
        self.b = np.zeros((1, d2))

    def forward(self, x):
        self.input = x
        return np.dot(x, self.w) + self.b
    
    def backward(self, delta, rate):
        # compute gradients
        dw = np.dot(self.input.T, delta)
        db = np.sum(delta, axis=0, keepdims=True)

        self.w -= rate * dw
        self.b -= rate * db

        # return gradient for previous layer
        return np.dot(delta, self.w.T)
    
def relu(x):
    return np.maximum(0, x)

def d_relu(x):  # derivative of relu
    return (x > 0).astype(float)

class MLP:
    def __init__(self, input_dim, dimensions) -> None:
        self.layers = []
        for d in dimensions:
            self.layers.append(Linear(input_dim, d))
            input_dim = d
    
    def forward(self, x):
        for layer in self.layers[:-1]:  # apply relu for hidden layers (not on output)
            x = relu(layer.forward(x))
        return self.layers[-1].forward(x)   # output layer does not need activation function
    
    def backward(self, y, y_pred, rate):
        delta = (y_pred - y) / y.shape[0]   # normalize by number of samples
        # print(f'1: {delta}')
        for layer in reversed(self.layers):
            # print(f'2: {delta}')
            if layer != self.layers[-1]:    # apply relu derivative for hidden layers (not on output)
                # print(layer.input)
                # print(f'3: {delta}')
                delta *= d_relu(layer.input)
            delta = layer.backward(delta, rate)
            # print(f'4: {delta}')

    def train(self, X, y, epochs, rate):
        for epoch in range(epochs):
            # print(f'epoch: {epoch}')
            y_pred = self.forward(X)
            # binary cross-entropy loss
            loss = -np.mean(y * np.log(y_pred + 1e-12) + (1 - y) * np.log(1 - y_pred + 1e-12))
            self.backward(y, y_pred, rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')



if __name__ == "__main__":
    X = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])

    y = np.array([[0], [1], [0], [1], [0]])

    mlp = MLP(input_dim=4, dimensions=[3, 3, 1])
    mlp.train(X, y, epochs=1000, rate=0.1)
    predictions = mlp.forward(X)
    print("Prediction: \n", predictions)