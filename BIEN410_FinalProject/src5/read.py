import numpy as np

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


def read(file, win):
    # input: labels.txt     window
    # output: X, y
    if win % 2 == 0: print('warning: window size must be odd')
    data = {}    
    with open(file, 'r') as f:
        while True:
            id = f.readline().strip()
            if not id: break
            seq = f.readline().strip()
            label = f.readline().strip()

            p = Protein(id, seq, win)
            p.label = label
            data[p.id] = p
    
    X = []
    y = []
    for p in data.values():
        for i in range(len(p)):
            # update X
            x = np.zeros(20*win)
            for j in range(win):
                if p.seq_flank[j+i] == '-': continue
                x[20*j+symToNum(p.seq_flank[j+i])] = 1

            # start = max(0, i-win//2)
            # end = min(len(p), i+1+win//2)
            # for j in range(start, end):
            #     r = symToNum(p.seq[j])
                
            X.append(x)

            # update y
            if p.label[i] == 'H': 
                y.append(1)
            else:
                y.append(0) 
    X = np.array(X)
    y = np.array(y)
    # print(X)
    # print(X.shape)
    # print(y)
    # print(y.shape)
    return X, y, data

# read('../training_data/labels.txt', 3)

