import cvxpy as cp
import numpy as np
import torch
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood
from stpy.probability.gaussian_likelihood import GaussianLikelihood

class LaplaceLikelihood(GaussianLikelihood):

    def __init__(self, sigma = 0.1, Sigma=None):
        super().__init__()
        self.sigma = sigma
        self.Sigma = Sigma

    def evaluate_log(self, f):
        if self.Sigma is None:
            res = torch.sum(torch.abs(f - self.y))/self.sigma
        else:
            res = torch.sum(np.abs(torch.inverse(self.Sigma) @ (f - self.y)))
        return res

    def evaluate_point(self, theta, d):
        x, y = d
        if self.Sigma is None:
            return (torch.abs(x @ theta - y)) / self.sigma
        else:
            return (x @ theta - y).T @ torch.linalg.inv(self.Sigma.T @ self.Sigma) @ (
                    x @ theta - y)

    def get_cvxpy_objective(self, mask = None):
        if mask is None:
            if self.Sigma is None:
                def likelihood(theta): return cp.sum(cp.abs(self.x@theta - self.y)/self.sigma)

            else:
                def likelihood(theta): return cp.sum(cp.abs(torch.linalg.inv(self.Sigma)@(self.x@theta - self.y)))
        else:
            if self.Sigma is None:
                def likelihood(theta):
                    if torch.sum(mask.int())>0:
                        return cp.sum(cp.abs(self.x[mask,:]@theta - self.y[mask,:])/self.sigma)
                    else:
                        return cp.sum(theta*0)
            else:
                def likelihood(theta):
                    if torch.sum(mask.int())>0:
                        return cp.sum(cp.abs(torch.linalg.inv(self.Sigma)@(self.x[mask,:]@theta - self.y[mask,:])))
                    else:
                        return cp.sum(theta*0)
        return likelihood


    def information_matrix(self):
        if self.Sigma is None:
            V = self.x.T@self.x/self.sigma
        else:
            V = self.x.T@self.Sigma.T@self.x
        return V


    def get_torch_objective(self):
        raise NotImplementedError("Implement me please.")
