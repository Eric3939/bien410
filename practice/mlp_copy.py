import numpy as np
import matplotlib.pyplot as plt

class Linear:
    def __init__(self, in_features, out_features):
        # Initialize weights and biases with small random values
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))

    def forward(self, x):
        # Linear transformation: Wx + b
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        # Gradient w.r.t. weights and biases
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        # Gradient w.r.t. input to pass backward
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input


class MLP:
    def __init__(self, input_size, layer_sizes, output_size):
        # Create a list of layers with the given sizes
        sizes = [input_size] + layer_sizes + [output_size]
        self.layers = []
        self.loss = []

        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1]))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        # Pass input through each layer
        self.activations = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            # Apply ReLU activation except for the last layer
            if i < len(self.layers) - 1:
                x = self.relu(x)
            self.activations.append(x)
        return x

    def backward(self, grad_output, learning_rate):
        # Backpropagation through each layer
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:  # Not the last layer
                grad_output *= self.relu_derivative(self.activations[i])
            grad_output = self.layers[i].backward(grad_output, learning_rate)

    def train(self, x, y, learning_rate, epochs):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(x)
            # Compute the loss (Mean Squared Error)
            loss = np.mean((y_pred - y) ** 2)
            self.loss.append(loss)
            # Backward pass
            grad_output = 2 * (y_pred - y) / y.shape[0]  # Derivative of MSE loss
            self.backward(grad_output, learning_rate)



if __name__ == "__main__":
    X = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 0, 0]
    ])

    y = np.array([[0], [1], [0], [1], [0]])

    mlp = MLP(4, [3, 2], 1)
    mlp.train(X, y,  0.1, 10000)
    predictions = mlp.forward(X)
    print("Prediction: \n", predictions)
    plt.figure()
    plt.scatter(range(len((mlp.loss))), mlp.loss, s=3)
    plt.savefig('plot.png')