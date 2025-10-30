class MLP:
    def __init__ (self, input, hidden, output=1):
        self.dA = []
        self.layers = []
        layers = [input] + hidden + [output]     # number of neurons in each layer
        for i in range(len(layers) - 1):
            n, m = layers[i], layers[i + 1]
            if i == len(layers) - 2:
                # normalized xavier weight initialization for sigmoid       Glorot et al. 2010
                lower, upper =-(sqrt(6.0) / sqrt(n + m)), (sqrt(6.0) / sqrt(n + m))
                w = lower + (upper - lower) * np.random.rand(n, m)    
            else:
                # he weight initialization for relu      He et al. 2015
                std = sqrt(2.0 / n)
                w = std * np.random.randn(n, m)
            b = np.zeros((1, layers[i + 1]))
            self.layers.append((w, b))

    def relu(self, x):
        return np.maximum(0, x)
    
    def d_relu(self, x):
        return x > 0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.cache = []     # save the linear combination (Z) for backpropagation
        A = X
        for i, (W, b) in enumerate(self.layers):
            Z = A @ W + b
            self.cache.append((A, Z))
            if i < len(self.layers) - 1:
                A = self.relu(Z)        # relu for hidden layers
            else:
                A = self.sigmoid(Z)     # sigmoid for output layer
        return A
    
    def backward(self, X, y, rate=0.1):
        m = y.shape[0]      # number of samples
        dA = self.forward(X) - y.reshape(-1, 1)     # derivative of binary cross-entropy
        self.dA.append(dA.mean())
        for i in reversed(range(len(self.layers))):
            A_prev, Z = self.cache[i]
            W, b = self.layers[i]
            if i < len(self.layers) - 1:        # backpropagation (it's just a chain rule)
                dZ = dA * self.d_relu(Z)
            else:
                dZ = dA
            dW = A_prev.T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA = dZ @ W.T
            self.layers[i] = (W - rate * dW, b - rate * db)     # gradient descent

    def fit(self, X, y, epochs=1000, rate=0.1):
        n = 10
        i = 0
        for _ in range(epochs):
            self.backward(X, y, rate)
            if abs(self.dA[-1]) < 1e-4:
                n -= 1 
            if n <= 0 : break   # training stops when consecutive low dA
            i += 1
        return i
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)
    
    def save_parameters(self, filename):
        with open(filename, 'w') as f:
            for layer in self.layers:
                np.savetxt(f, layer[0].flatten(), header="Weights of Layer")
                np.savetxt(f, layer[1].flatten(), header="Bias of Layer")
            f.write("#")

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
                        weight = l.reshape(layer[0].shape)    # weight
                    else:
                        bias = l.reshape(layer[1].shape)    # bias
                        layer = (weight, bias)
                        self.layers[i//2] = layer
                    l = []
                    i+=1
                    continue
                l.append(np.float64(line))    

