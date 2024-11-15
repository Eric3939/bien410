# Naive Base

import numpy as np

class Protein:
    def __init__(self, id, seq) -> None:
        self.id = id
        self.seq = seq
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

def main():
    # read
    data = {}    
    inputFile = '../training_data/labels.txt'
    with open(inputFile, 'r') as f:
        while True:
            id = f.readline().strip()
            if not id: break
            seq = f.readline().strip()
            label = f.readline().strip()

            p = Protein(id, seq)
            p.label = label
            data[p.id] = p
    

    # build parameters
    win = 7
    if win % 2 == 0: print('warning: window size must be odd')
    H1 = 0
    H0 = 0
    cp = np.zeros((20, win, 2))      # conditional probabilities matrix: row: 20 amino acids   columns: window of size      depth: H=1, H=0
    for p in data.values():
        for i in range(len(p)):
            # update H1, H0
            if p.label[i] == 'H':
                H1 += 1
            else:
                H0 += 1
            # update cp matrix
            start = max(0, i-win//2)
            end = min(len(p), i+1+win//2)
            for j in range(start, end):
                r = symToNum(p.seq[j])
                col = j-i+win//2        # calculates which column in cp matrix to update
                if p.label[j] == 'H':
                    cp[r][col][0] += 1
                else:
                    cp[r][col][1] += 1
    cp[:, :, 0] /= H1
    cp[:, :, 1] /= H0
    total = H1 + H0
    H1 = H1 / total
    H0 = H0 / total

    # write

    with open('parameters.txt', 'w') as f:
        f.write(f'{H1}\n')
        f.write(f'{H0}\n')
        for i in range(len(cp)):
            for j in range(len(cp[0])):
                for k in range(len(cp[0][0])):
                    f.write(f'{cp[i][j][k]}\n')



if __name__ == '__main__':
    main()