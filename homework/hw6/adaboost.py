import numpy as np
from utils import KNNClassifier, Perceptron


class AdaBoost:
    def __init__(self, baseline_num):
        self.baselines = []
        self.alphas = []
        self.T = baseline_num

    def train(self, dataset):
        X_train, y_train = dataset.get_dataset()
        D = np.ones(y_train.shape) / y_train.size

        for t in range(self.T):
            X, y = dataset.get_subset(D)
            baseline = KNNClassifier(X, y, 3)
            alpha = self.update_D(X_train, y_train, baseline, D)

            self.baselines.append(baseline)
            self.alphas.append(alpha)

    def update_D(self, X_train, y_train, baseline, D):
        h = np.array([baseline.predict(x) for x in X_train])

        epsilon = D @ (h != y_train).astype(int)
        if epsilon != 0.0:
            alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        else:
            alpha = 0.0

        sign = 2 * (y_train == h) - 1
        Z = D * np.exp(sign * alpha)
        D = Z / Z.sum()
        # print(f'Alpha: {alpha}, D: {D}')

        return alpha

    def predict(self, x):
        h = self.baseline_predict(x)
        alpha = np.float32(self.alphas)

        return (alpha @ h > 0.0)

    def baseline_predict(self, x):
        h = np.float32([baseline.predict(x) for baseline in self.baselines])
        return h


class AdaBoostTextbook(AdaBoost):
    def __init__(self, baseline_num):
        super().__init__(baseline_num)

    def train(self, dataset, eta, epochs):
        super().train(dataset)
        X_train, y_train = dataset.get_dataset()
        weight_learner = Perceptron(self.T, 0.2, 0.2)

        for epoch in range(epochs):
            for x, y in zip(X_train, y_train):
                h = super().baseline_predict(x)
                h = np.hstack([1, h])
                H = weight_learner.forward(h)
                weight_learner.update(h, y)

            h_list = []
            for x in X_train:
                h_list.append(super().baseline_predict(x))
            h = np.hstack([np.ones((y_train.size, 1)), np.vstack(h_list)])
            H = weight_learner.forward(h)
            if np.all(H == y_train):
                break
        else:
            print(f'Exceeded maximum {epochs} epochs. Training terminated.')
            # print(weight_learner.w)
