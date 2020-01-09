import numpy as np
from adaboost import AdaBoost, AdaBoostTextbook
from utils import Dataset


def test(model, dataset, name):
    X_test, y_test = dataset.get_dataset()
    pred = np.array([model.predict(x) for x in X_test])
    accuracy = (y_test == pred).sum() / y_test.size
    print(f'{name} version accuracy: {accuracy:.1f}')


if __name__ == '__main__':
    dataset = Dataset('./training-data.txt')
    test_dataset = Dataset('./testing-data.txt')

    model = AdaBoost(9)
    model.train(dataset)
    accuracy = test(model, test_dataset, 'Original')

    model_tb = AdaBoostTextbook(9)
    model_tb.train(dataset, 0.2, 2)
    accuracy_tb = test(model_tb, test_dataset, 'Textbook')
