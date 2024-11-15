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
    H1 = 0
    H0 = 0
    cp = np.zeros((2, 20))      # conditional probabilities matrix: row: H=1, H=0   columns: 20 amino acids
    for p in data.values():
        for i in range(len(p)):
            r = symToNum(p.seq[i])
            if p.label[i] == 'H':
                H1 += 1
                cp[0][r] += 1
            else:
                H0 += 1
                cp[1][r] += 1

    cp[0] = cp[0]/H1
    cp[1] = cp[1]/H0
    total = H1 + H0
    H1 = H1 / total
    H0 = H0 / total

    # write

    with open('parameters.txt', 'w') as f:
        f.write(f'{H1}\n')
        f.write(f'{H0}\n')
        for i in range(len(cp)):
            for j in range(len(cp[0])):
                f.write(f'{cp[i][j]}\n')



if __name__ == '__main__':
    main()