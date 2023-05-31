import cvxpy as cp
import numpy as np
import torch
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood
from stpy.probability.gaussian_likelihood import GaussianLikelihood


class HuberLikelihood(GaussianLikelihood):

    def __init__(self, sigma=0.1, M=1.):
        super().__init__()
        self.sigma = sigma
        self.M = M

    def evaluate_log(self, f):
        pass

    def scale(self, err = None):
        if self.Sigma is None:
            return self.sigma**2
        else:
            return torch.max(self.Sigma.T@self.Sigma)

    def evaluate_datapoint(self, theta, d, mask = None):
        if mask is None:
            mask = 1.
        x, y = d
        res = (x @ theta - y) / self.sigma
        mask1 = torch.abs(res) < self.M
        mask2 = torch.abs(res) >= self.M
        v = res
        v[mask1] = res[mask1] ** 2
        v[mask2] = 2 * self.M * torch.abs(res[mask2]) - self.M ** 2
        return torch.sum(v)*mask

    def add_data_point(self, d):
        x, y = d
        self.x = torch.vstack(self.x, x)
        self.y = torch.vstack(self.y, y)
        self.fitted = False

    def load_data(self, D):
        self.x, self.y = D
        self.fitted = False

    def get_objective_cvxpy(self, mask=None):
        if mask is None:
            def likelihood(theta):
                return cp.sum(cp.huber((self.x @ theta - self.y) / self.sigma))
        else:
            def likelihood(theta):
                if torch.sum(mask.int()) > 0:
                    return cp.sum(cp.huber((self.x[mask, :] @ theta - self.y[mask, :]) / self.sigma))
                else:
                    return cp.sum(theta * 0)
        return likelihood

    def information_matrix(self):
        V = self.x.T @ self.x / self.sigma
        return V


    def get_objective_torch(self):
        raise NotImplementedError("Implement me please.")
