import sys
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

def load_data(filename):
    '''Loads data, formats it, and returns a training set and a testing set.'''
    data = pd.read_csv(filename, header=None)

    group = dict(list(data.groupby(data.iloc[:, -1])))

    t_size = 30
    t_list = map(lambda key: group[key][:t_size], [*group])
    d_list = map(lambda key: group[key][t_size:], [*group])

    tset = pd.concat(t_list)
    dset = pd.concat(d_list)

    return [tset, dset]

def k_nn(tset, k, dset):
    '''Executes k-NN algorithm and returns accuracy information.'''
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(tset.iloc[:, :-1], tset.iloc[:, -1])

    example = np.array(dset.iloc[:, :-1]).reshape(len(dset), -1)
    predict = model.predict(example)
    
    d_size = len(predict)
    success = 0
    
    for p, x in zip(predict, dset.iloc[:, -1]):
        if p == x:
            success += 1

    return [success, d_size - success, success / d_size]

TSET, DSET = load_data(sys.argv[1])
SUCC, FAIL, ACCU = k_nn(TSET, 1, DSET)
print(f'Success: {SUCC}\nFailure: {FAIL}\nAccuracy: {ACCU:.3f}')
