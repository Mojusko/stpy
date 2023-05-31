import cvxpy as cp
import numpy as np
import torch
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood
from typing import Union, Dict, List


class WeilbullLikelihoodCanonical(Likelihood):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def information_matrix(self, theta_fit):
        pass


    def normalization(self, d):
        pass


    def evaluate_datapoint(self, theta, d, mask = None):
        if mask is None:
            mask = 1.
        x, y = d
        lam = torch.exp(x @ theta)
        l = -torch.log(lam) + (y ** (self.p)) * lam
        l = l * mask
        return l

    def scale(self, err = None, bound = None):
        return np.exp(bound)

    def add_data_point(self, d):
        x, y = d
        self.x = torch.vstack(self.x, x)
        self.y = torch.vstack(self.y, y)
        self.fitted = False

    def load_data(self, D):
        self.x, self.y = D
        self.fitted = False

    def evaluate_log(self, f):
        pass

    def get_objective_torch(self):
        pass

    def get_objective_cvxpy(self, mask=None):
        if mask is None:
            def likelihood(theta):
                return -cp.sum(self.x@theta) + cp.sum(cp.diag(self.y**(self.p))@cp.exp(self.x @ theta))
        else:
            def likelihood(theta):
                if torch.sum(mask.int())>0:
                    return - cp.sum(self.x[mask,:] @ theta) + cp.sum(cp.diag(self.y[mask,:]**(self.p))@cp.exp(self.x[mask,:] @ theta))
                else:
                    return cp.sum(theta * 0)
        return likelihood

    def get_confidence_set_cvxpy(self,
                                 theta: cp.Variable,
                                 type: Union[str, None] = None,
                                 params: Dict = {},
                                 delta: float = 0.1):
        if self.fitted == True:
            return self.set_fn(theta)

        theta_fit = params['estimate']
        H = params['regularizer_hessian']

        beta = self.confidence_parameter(delta, params, type=type)

        if type in ["laplace"]:
            V = self.information_matrix(theta_fit)
            if H is not None:
                 V += H
            self.set_fn = lambda theta: [cp.quad_form(theta - theta_fit, V) <= beta]
            set = self.set_fn(theta)

        elif type == "LR":
            set = self.lr_confidence_set_cvxpy(theta, beta, params)

        else:
            raise NotImplementedError("The desired confidence set type is not supported.")

        self.set = set
        self.fitted = True
        return set

    def confidence_parameter(self, delta, params, type = None):
        if type == "LR":
            # this is based on sequential LR test
            beta = self.confidence_parameter_likelihood_ratio(delta, params)
        elif type == "laplace":
            beta = 2.
        else:
            raise NotImplementedError("The desired confidence set type is not supported.")
        return beta