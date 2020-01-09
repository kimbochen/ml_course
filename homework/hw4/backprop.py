import numpy as np
import pandas as pd
# from numpy.random import uniform


class MultiLayerPerceptron:
    def __init__(self, x, y, eta):
        self.m, self.n = x.shape
        self.eta = eta
        # self.w1 = uniform(-0.1, 0.1, (self.n, self.n))
        # self.w2 = uniform(-0.1, 0.1, (np.unique(y).size, self.n))
        self.w1 = np.float32([[-1.0, 1.0],
                              [1.0, 1.0]])
        self.w2 = np.float32([[1.0, 1.0],
                              [-1.0, 1.0]])

    def sigmoid(self, z):
        z = np.array(z)
        g = 1 / (1 + np.exp(-z))

        return g

    def forward(self, x):
        self.a2 = self.sigmoid(self.w1 @ x)
        h = self.sigmoid(self.w2 @ self.a2)

        return h

    def backward(self, x, y, h):
        delta2 = h * (1 - h) * (y - h)
        delta1 = self.a2 * (1 - self.a2) * (self.w1.T @ delta2)

        self.w2 = self.w2 + self.eta * delta2 * self.a2.T
        self.w1 = self.w1 + self.eta * delta1 * x.T
        # print(delta1, x.T, sep='\n\n')


names = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'class']

df = pd.read_csv('iris.data.txt', header=None, names=names)

# X = np.float32(df.drop('class', axis=1))
# X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
# X.shape = (150, 5)

# y = np.float32(pd.factorize(df['class'])[0])
# y.shape = (150, )

a = np.array([[1],
              [2]])
b = np.array([[3],
              [4]])

# print(a * b.T)

X = np.float32([[1],
                [-1]])

y = np.float32([[1],
                [0]])

model = MultiLayerPerceptron(X, y, 0.1)

h = model.forward(X)
model.backward(X, y, h)

# print(model.w1, '\n\n', model.w2)
