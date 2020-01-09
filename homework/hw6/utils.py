import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, file_path):
        np.random.seed(9)
        columns = ['a', 'b', 'c', 'd', 'label']
        df = pd.read_csv(file_path, header=None, names=columns)
        df['label'] = pd.factorize(df['label'])[0]
        self.dataframe = df

    def get_dataset(self):
        X, y = self.df_to_np(self.dataframe)
        return X, y

    def get_subset(self, D):
        data_subset = self.dataframe.sample(frac=0.1, replace=True, weights=D)
        return self.df_to_np(data_subset)

    def df_to_np(self, df):
        X = np.float32(df.drop('label', axis=1))
        y = np.array(df['label'])

        return X, y


class KNNClassifier:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
        # print(f'Baseline Classifier:\nX:\n{self.X}\ny:\n{self.y}')

    def predict(self, e):
        dist = ((self.X - e * np.ones(self.X.shape)) ** 2).sum(axis=1)
        select = np.argpartition(dist, self.k)[0:self.k]
        label, freq = np.unique(self.y[select], return_counts=True)
        # print(f'Distance: {dist} Selection: {select} Pick: {freq.argmax()}')

        return label[freq.argmax()]


class Perceptron:
    def __init__(self, m, initial_weight, eta):
        self.w = np.ones(m+1) * initial_weight
        self.lr = eta

    def forward(self, x):
        self.h = (x @ self.w > 0.0).astype(int)
        return self.h

    def update(self, x, y):
        self.w += self.lr * (self.h - y) * x
        # print(self.w, self.h, y, x)
