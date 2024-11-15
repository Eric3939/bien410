# MLP

import numpy as np
from read import read
from time import time

class Protein:
    def __init__(self, id, seq, win) -> None:
        self.id = id
        self.seq = seq
        flank = win // 2
        self.seq_flank = '-'*flank + self.seq + '-'*flank
        self.label = None
        self.pred = None

    def __str__(self) -> str:
        return f'{self.id}\n{self.seq}\n{self.label}\n{self.pred}\n'

    def __len__(self) -> int:
        return len(self.seq)

class Linear:
    def __init__(self, d1, d2) -> None:     # d1: neurons in previous layer     d2: neurons in current layer
        self.w = np.random.randn(d1, d2) * 0.01     # initialize random weights
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
        with open('log.txt', 'w')as f:
            pass
        for epoch in range(epochs):
            y_pred = self.forward(x)
            # binary cross entropy loss
            loss = - np.mean(y * np.log(y_pred + 1e-12) + (1 - y) * np.log(1 - y_pred + 1e-12))
            self.loss.append(loss)
            delta = (y_pred - y) / y.shape[0]
            self.backward(delta, rate)
            # if epoch%100 == 0:
            print(f'epoch: {epoch}\tloss:{loss}')
            with open('log.txt', 'a') as f:
                f.write(f'epoch: {epoch}\tloss:{loss}'+'\n')
    
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

def symToNum(symbol):
    d = {'A':0, 'R':1, 'N':2, 'D':3, 
         'C':4, 'E':5, 'Q':6, 'G':7, 
         'H':8, 'I':9, 'L':10, 'K':11, 
         'M':12, 'F':13, 'P':14, 'S':15, 
         'T':16, 'W':17, 'Y':18, 'V':19}
    return d[symbol]

def main():
    # read
    t1 = time()
    inputFile = '../training_data/labels.txt'
    X, y, data = read(inputFile, 9)
    y = y.reshape((-1, 1))
    t2 = time()
    print(f'read completed. time: {round(t2-t1)}s')

    # MLP
    t1 = time()
    mlp = MLP(X.shape[1], [90, 45, 20, 10], 1)
    # mlp.train(X, y,  0.1, 300)
    # mlp.save_parameters('parameters.txt')
    mlp.load_parameters('parameters.txt')
    predictions = mlp.forward(X)
    predictions = (predictions >= 0.5).astype(int)
    print(predictions)
    t2 = time()
    print(f'MLP completed. time: {round(t2-t1)}s')
    n = 0
    for p in data.values():
        pred = ''
        for _ in range(len(p)):
            if predictions[n] == 1:
                pred += 'H'
            else:
                pred += '-'
            n+=1
        p.pred = pred
    print('prediction updated')

    # write
    with open('outfile.txt', 'w') as f:
        for p in data.values():
            f.write(f'{p.id}\n')
            f.write(f'{p.seq}\n')
            f.write(f'{p.pred}\n')
    print('write completed')



if __name__ == '__main__':
    main()