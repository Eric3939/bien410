# MLP

import numpy as np
from math import sqrt
from read import read
from time import time
from datetime import datetime
import subprocess


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

def symToNum(symbol):
    d = {'A':0, 'R':1, 'N':2, 'D':3, 
         'C':4, 'E':5, 'Q':6, 'G':7, 
         'H':8, 'I':9, 'L':10, 'K':11, 
         'M':12, 'F':13, 'P':14, 'S':15, 
         'T':16, 'W':17, 'Y':18, 'V':19}
    return d[symbol]

def out(predictions, data, outfile):
    # output the predictions to outfile.txt
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

    # write
    with open(outfile, 'w') as f:
        for p in data.values():
            f.write(f'{p.id}\n')
            f.write(f'{p.seq}\n')
            f.write(f'{p.pred}\n')

def accuracy():
    result = subprocess.run(['python3', '../testing/testing.py', '-p', 'outfile.txt', '-l', '../training_data/labels.txt'], capture_output=True, text=True)
    result = result.stdout
    result = result.split()[2]
    return float(result)

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



def main():
    curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f'log_{curr_time}.txt', 'w') as f:
        f.write('')

    # read
    t1 = time()
    inputFile = '../training_data/labels.txt'
    X, y, data = read(inputFile, 9)
    y = y.reshape((-1, 1))
    t2 = time()
    print(f'read completed. time: {round(t2-t1)}s')

    structure = [
        [90, 45, 20, 10],
        [60, 120, 100, 50],
        [128, 64, 32, 16, 8],
        [30, 90, 120, 90, 30],
        [240, 180, 90, 40, 20],
        [300, 180, 90, 30]
    ]

    rates = [
        0.1, 
        0.5
    ]

    # MLP training
    best_acc = 0
    for stru in structure:
        for rate in rates:
            for _ in range(1):
                mlp = MLP(X.shape[1], stru, 1)
                iter = mlp.fit(X, y, 10000, rate)
                predictions = mlp.predict(X)
                out(predictions, data, 'outfile.txt')
                acc = accuracy()
                if acc > best_acc:
                    best_acc = acc
                    mlp.save_parameters(f'best_para_{curr_time}.txt')
                
                # write log
                with open(f'log_{curr_time}.txt', 'a') as f:
                    f.write(f'{acc}\t{rate}\t{iter}\t{str(stru)}\n')


                






if __name__ == '__main__':
    main()