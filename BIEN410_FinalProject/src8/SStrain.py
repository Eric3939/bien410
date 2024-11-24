# MLP pytorch

import numpy as np
from math import sqrt
from read import read
from time import time
from datetime import datetime
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim

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

class MLP(nn.Module):
    def __init__(self, input_dim, layers):
        super(MLP, self).__init__()
        self.model = self._build_model(input_dim, layers)
    
    def _build_model(self, input_dim, layers):
        modules = []
        for layer_dim in layers[:-1]:
            modules.append(nn.Linear(input_dim, layer_dim))
            modules.append(nn.ReLU())
            input_dim = layer_dim
        modules.append(nn.Linear(input_dim, layers[-1]))
        modules.append(nn.Sigmoid())  # For binary classification
        return nn.Sequential(*modules)

    def _check_device(self):
        if torch.cuda.is_available():
            print("Current device:", torch.cuda.current_device())
            print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        else:
            print("No GPU detected, using CPU.")

    def forward(self, x):
        return self.model(x)

    def fit(self, X, y, epochs=1000, lr=0.01):      # input: numpy
        print("fitting...")
        self._check_device()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)

        loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            # forward
            y_pred = self(X_tensor)
            loss = loss_fn(y_pred, y_tensor)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 500 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    def predict(self, X):       # input numpy
        print("predicting...")
        self._check_device()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)        
        X_tensor = torch.from_numpy(X).float().to(device)
        with torch.no_grad():
            y_pred = self(X_tensor).cpu().numpy()
        
        return (y_pred > 0.5).astype(int)


    def save_parameters(self, filename):
        with open(filename, "w") as f:
            for name, param in self.named_parameters():
                param = param.data.cpu().numpy()
                param = param.T
                if "weight" in name:
                    np.savetxt(f, param.flatten(), header=f"Weights of Layer {name}")
                elif "bias" in name:
                    np.savetxt(f, param.flatten(), header=f"Bias of Layer {name}")
                else:
                    np.savetxt(f, param.flatten(), header=f"Unknown {name}")
            f.write("#")
        print(f"Parameters saved to {filename}")


def main():
    curr_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open(f'log_{curr_time}.txt', 'w') as f:
        f.write('')

    # read
    t1 = time()
    inputFile = '../training_data/labels.txt'
    window = 9
    X, y, data = read(inputFile, window)
    y = y.reshape((-1, 1))
    t2 = time()
    print(f'read completed. time: {round(t2-t1)}s')

    structure = [
        [160, 140, 100, 80, 51, 34, 17, 4, 1]
    ]

    rates = [
        0.01, 
        0.005
    ]

    # MLP training
    best_acc = 0
    for stru in structure:
        for rate in rates:
            for _ in range(3):
                t1 = time()
                print(f"mlp {stru} {rate} starts")
                mlp = MLP(X.shape[1], stru)
                iter = mlp.fit(X, y, 10000, rate)
                predictions = mlp.predict(X)
                out(predictions, data, 'outfile.txt')
                acc = accuracy()
                if acc > best_acc:
                    best_acc = acc
                    mlp.save_parameters(f'best_para_{curr_time}.txt')
                t2 = time()

                # write log
                with open(f'log_{curr_time}.txt', 'a') as f:
                    f.write(f'{acc}\t{rate}\t{iter}\t{window}win\t{str(stru)}\t{round((t2-t1)//60)}min\n')
                print(f"mlp ends\n\n\n")


                






if __name__ == '__main__':
    main()