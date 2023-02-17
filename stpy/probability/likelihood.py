from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np
import torch

class Likelihood(ABC):

    def __init__(self):
        self.fitted = False
        pass

    def evaluate_log(self, f):
        pass

    @abstractmethod
    def scale(self):
        return

    @abstractmethod
    def evaluate_point(self, f, d):
        pass

    def add_data_point(self, d):
        x,y = d
        self.x = torch.vstack(self.x,x)
        self.y = torch.vstack(self.y,y)
        self.fitted = False

    def load_data(self, D):
        self.x, self.y = D
        self.fitted = False

    @abstractmethod
    def get_cvxpy_objective(self, mask = None):
        pass

    @abstractmethod
    def get_torch_objective(self):
        pass

    def confidence_parameter_likelihood_ratio(self, delta, params):
        evidence = params['evidence']
        estimators = params['estimator_sequence']
        val = 0.


        for i in range(len(estimators)-1):
            ev = evidence[i+1]
            if ev is True:
                est = estimators[i]
                if est is not None:
                    val += self.evaluate_point(est, (self.x[i+1,:].view(1,-1), self.y[i+1,:].view(1,-1)))

        val = np.log(1/delta) + val

        return val

    def lr_confidence_set(self, theta, beta, params):
        evidence = torch.Tensor(params['evidence']).bool()
        self.set_fn = lambda theta:  [self.get_cvxpy_objective(mask = evidence)(theta) <= beta]
        set = self.set_fn(theta)
        return set


