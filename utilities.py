import numpy as np


def load_data(direc, ratio, dataset):
    """Input:
    direc: location of the UCR archive
    ratio: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')
    DATA = np.concatenate((data_train, data_test_val), axis=0)
    N = DATA.shape[0]
    ratio = (ratio * N).astype(np.int32)
    ind = np.random.permutation(N)
    df_train = DATA[ind[:ratio[0]]]
    df_val = DATA[ind[ratio[0]:ratio[1]]]
    df_test = DATA[ind[ratio[1]:]]
    return df_train, df_val, df_test


class DataIterator():
    def __init__(self, df, seq_len, n_inputs):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()
        self.seq_len = seq_len
        self.n_inputs = n_inputs

    def shuffle(self):
        np.random.shuffle(self.df)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor + n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df[self.cursor:self.cursor + n]
        self.cursor += n
        # Targets have labels 1-indexed. We subtract one for 0-indexed
        return res[:, 1:].reshape((-1, self.seq_len, self.n_inputs)), res[:, 0] - 1
