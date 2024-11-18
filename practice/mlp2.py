import numpy as np
import matplotlib.pyplot as plt


class Linear:
    def __init__(self, d1, d2) -> None:     # d1: neurons in previous layer     d2: neurons in current layer
        self.w = np.random.randn(d1, d2) * 0.1     # initialize random weights
        self.b = np.zeros((1, d2))

    def forward(self, x):
        self.input = x      # for later backpropagation
        return np.dot(x, self.w) + self.b
    
    def backward(self, delta, rate):
        # backpropagation
        dw = np.dot(self.input.T, delta)
        db = np.sum(delta, axis=0, keepdims=True)
        delta = np.dot(delta, self.w.T)     # update delta by mutliplying correponding weight, then pass it to the previous layer

        # gradient descent
        self.w -= rate * dw
        self.b -= rate * db

        return delta

class MLP:
    def __init__(self, input, layer, output) -> None:
        sizes = [input] + layer + [output]      # number of neurons
        self.layers = []
        self.loss = []

        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1]))

    def relu(self, x):
        return np.maximum(0, x)
    
    def d_relu(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        self.activations = []   # value after activation
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            # apply relu activation except for the last layer
            if i < len(self.layers) - 1:
                x = self.relu(x)
            self.activations.append(x)
        return x
    
    def backward(self, delta, rate):
        for i in reversed(range(len(self.layers))):
            # chain rule except for the last layer
            if i < len(self.layers) - 1:
                delta *= self.d_relu(self.activations[i])
            delta = self.layers[i].backward(delta, rate)

    def train(self, x, y, rate, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            # binary cross entropy loss
            loss = - np.mean(y * np.log(y_pred + 1e-12) + (1 - y) * np.log(1 - y_pred + 1e-12))
            self.loss.append(loss)
            delta = (y_pred - y) / y.shape[0]
            self.backward(delta, rate)
            if epoch%100 == 0:
                print(f'epoch: {epoch}\tloss:{loss}')
    
    def save_parameters(self, filename):
        with open(filename, 'w') as f:
            for layer in self.layers:
                np.savetxt(f, layer.w.flatten(), header="Weights of Layer")
                np.savetxt(f, layer.b.flatten(), header="Bias of Layer")

    def load_parameters(self, filename):
        with open(filename, 'r') as f:
            l = []
            next(f)
            i=0
            for line in f:
                if line[0] == "#":
                    l = np.array(l)
                    layer = self.layers[i//2]
                    if i%2 == 0:
                        layer.w = l.reshape(layer.w.shape)
                    else:
                        layer.b = l.reshape(layer.b.shape)
                    l = []
                    i+=1
                    continue
                l.append(float(line))

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=(0.8, 0.2), random_state=42)
    y = y.reshape((-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    mlp = MLP(X.shape[1], [10, 5], 1)
    mlp.train(X_train, y_train,  0.5, 10000)
    mlp.save_parameters('parameters.txt')
    # mlp.load_parameters('parameters.txt')
    predictions = mlp.forward(X_test)
    print(predictions)
    predictions = (predictions >= 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, predictions))

    plt.figure()
    plt.scatter(range(len((mlp.loss))), mlp.loss, s=1)
    plt.savefig('plot.png')