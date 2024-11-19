# Python 3 script to edit for this project. 
# Note 1: Do not change the name of this file
# Note 2: Do not change the location of this file within the BIEN410_FinalProject package
# Note 3: This file can only read in "../input_file/input_file.txt" and "parameters.txt" as input
# Note 4: This file should write output to "../output_file/outfile.txt"
# Note 5: See example of a working SSPred.py file in ../src_example folder

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
    # read input
    data = {}    
    inputFile = '../input_file/infile.txt'
    with open(inputFile, 'r') as f:
        while True:
            id = f.readline().strip()
            if not id: break
            seq = f.readline().strip()

            p = Protein(id, seq)
            data[p.id] = p

    # read parameters
    win = 9
    cp = np.zeros((20, win, 2))      # conditional probabilities matrix: row: 20 amino acids   columns: window of size      depth: H=1, H=0
    with open('parameters.txt', 'r') as f:
        H1 = float(f.readline().strip())
        H0 = float(f.readline().strip())
        for i in range(len(cp)):
            for j in range(len(cp[0])):
                for k in range(len(cp[0][0])):
                    cp[i][j][k] = f.readline().strip()
        
    # predict
    pprob =(1-cp[:, :, 0])/(1-cp[:, :, 1])  # precomputed prob
    pprob = np.prod(pprob)
    pprob = pprob * H1 / H0

    # mprob = np.full((20, win), pprob)       # prob matrix
    # for i in range(len(mprob)):
    #     for j in range(len(mprob[0])):
    #         add = cp[i][j] / cp[1][i]
    #         remove = (1-cp[0][i]) / (1-cp[1][i])
    #         mprob[i] = mprob[i] * add / remove

    for p in data.values():
        pred = ''
        for i in range(len(p)):
            start = max(0, i-win//2)
            end = min(len(p), i+1+win//2)
            score = pprob
            for j in range(start, end):
                r = symToNum(p.seq[j])
                col = j-i+win//2        # calculates which column in cp matrix
                # remove
                score /= (1-cp[r][col][0]) / (1-cp[r][col][1])
                # add
                score *= cp[r][col][0] / cp[r][col][1]
            if score >= 1:
                pred += 'H'
            else:
                pred += '-'
        p.pred = pred
    # write
    with open('outfile.txt', 'w') as f:
        for p in data.values():
            f.write(f'{p.id}\n')
            f.write(f'{p.seq}\n')
            f.write(f'{p.pred}\n')

if __name__ == '__main__':
    main()