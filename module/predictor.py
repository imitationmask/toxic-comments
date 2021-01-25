from abc import ABC, abstractmethod


class Predictor(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def predict(self):
        pass
