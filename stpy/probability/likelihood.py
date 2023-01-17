from abc import ABC, abstractmethod

class Likelihood(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, f):
        pass

    @abstractmethod
    def add_data_point(self, d):
        pass

    @abstractmethod
    def load_data(self, D):
        pass

    @abstractmethod
    def get_cvxpy_objective(self):
        pass

    @abstractmethod
    def get_torch_objective(self):
        pass