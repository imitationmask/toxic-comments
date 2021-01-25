from abc import ABC, abstractmethod


class Trainer(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def train_test_split(self, X, y, test_size, random_state):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
