# Naive Base

import numpy as np
from read import read
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier, MLPRegressor

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

class LogisticRegression:
    def __init__(self, rate = 0.001, iter=1000) -> None:
        self.rate = rate
        self.iter = iter
        self.w = None
        self.b = None
        self.loss = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)
    
    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        A = self._sigmoid(z)
        return A

    def fit(self, X, y):
        n_samples, n_feautres = X.shape

        # init parameters
        self.w = np.zeros(n_feautres)
        self.b = 0

        # gradient descent
        it = 0
        for _ in range(self.iter):
            A = self.forward(X)
            self.loss.append(self.compute_loss(y, A))
            dz = A - y  # derivative of sigmoid and bce X.T*(A-y)
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            # update parameters
            self.w -= self.rate * dw
            self.b -= self.rate * db
            it +=1
            print(f'\r{it}/{self.iter}', end='')

    def predict(self, X):
        thr = .5
        y_hat = np.dot(X, self.w) + self.b
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > thr else 0 for i in y_predicted]

        return np.array(y_predicted_cls)
    

def symToNum(symbol):
    d = {'A':0, 'R':1, 'N':2, 'D':3, 
         'C':4, 'E':5, 'Q':6, 'G':7, 
         'H':8, 'I':9, 'L':10, 'K':11, 
         'M':12, 'F':13, 'P':14, 'S':15, 
         'T':16, 'W':17, 'Y':18, 'V':19}
    return d[symbol]

def main():
    # read
    inputFile = '../training_data/labels.txt'
    X, y, data = read(inputFile, 9)
    print('read completed')
    # regressor = LogisticRegression(rate=0.0001, iter=1000)
    # regressor.fit(X, y)
    # predictions = regressor.predict(X)

    # regressor = LR(max_iter=200)
    # regressor.fit(X, y)
    # predictions = regressor.predict(X)


    # For classification
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=300, verbose=True)
    clf.fit(X, y)
    predictions = clf.predict(X)

    print('MLP completed')
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

    # # write

    # with open('parameters.txt', 'w') as f:
    #     f.write(f'{H1}\n')
    #     f.write(f'{H0}\n')
    #     for i in range(len(cp)):
    #         for j in range(len(cp[0])):
    #             for k in range(len(cp[0][0])):
    #                 f.write(f'{cp[i][j][k]}\n')



if __name__ == '__main__':
    main()