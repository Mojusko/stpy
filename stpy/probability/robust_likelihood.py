import cvxpy as cp
import numpy as np
import torch
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood

class RobustGraphicalLikelihood(Likelihood):

    def __init__(self, coin, supp,  sigma = 0.1):
        super().__init__()
        self.coin = coin
        self.supp = supp
        self.sigma = sigma

    def evaluate_log(self, f):
        pass

    def evaluate_point(self, theta, d):
        x, y = d
        return torch.log(1 + torch.exp())

    def add_data_point(self, d):
        x,y = d
        self.x = torch.vstack(self.x,x)
        self.y = torch.vstack(self.y,y)
        self.fitted = False

    def load_data(self, D):
        self.x, self.y = D
        self.fitted = False

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

    def get_cvxpy_confidence_set(self,
                                 theta: cp.Variable,
                                 type: Union[str, None] = None,
                                 params: Dict = {},
                                 delta: float = 0.1):
        if self.fitted == True:
            return self.set_fn(theta)

        theta_fit = params['estimate']
        H = params['regularizer_hessian']

        beta = self.confidence_parameter(delta, params, type=type)

        if H is not None:
            V = self.information_matrix() + H
        else:
            V = self.information_matrix()

        if type in ["none", None, "fixed", "adaptive-AB", "adaptive-optimized"]:
            self.set_fn = lambda theta: [cp.quad_form(theta - theta_fit, V) <= beta]
            set = self.set_fn(theta)

        elif type == "LR":
            set = self.lr_confidence_set(theta, beta, params)

        else:
            raise NotImplementedError("The desired confidence set type is not supported.")

        self.set = set
        self.fitted = True

        return set


    def get_torch_objective(self):
        raise NotImplementedError("Implement me please.")
