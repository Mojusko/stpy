import cvxpy as cp
import numpy as np
import torch
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood
from stpy.probability.gaussian_likelihood import GaussianLikelihood
import scipy

class BernoulliLikelihoodCanonical(GaussianLikelihood):

    def __init__(self):
        super().__init__()

    def evaluate_point(self, theta, d):
        x, y = d
        mu = 1./(1.+np.exp(-x@theta))
        r = -y*torch.log(mu) - (1-y)*torch.log(1-mu)
        return r

    def get_cvxpy_objective(self, mask = None):
        if mask is None:
            def likelihood(theta):
                return -self.y.T@(self.x @ theta) + cp.sum(cp.logistic(self.x @ theta))
        else:
            def likelihood(theta):
                if torch.sum(mask.int())>0:
                    return -self.y[mask,:].T@(self.x[mask,:] @ theta) + cp.sum(cp.logistic(self.x[mask,:] @ theta))
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


        # if H is not None:
        #     V = self.information_matrix() + H
        # else:
        #     V = self.information_matrix()

        if type in ['Faubry']:
            v = self.x @ theta_fit
            V = self.x.T @torch.diag(v) @ self.x + H
            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            beta = self.confidence_parameter(delta, params, type=type)
            self.set_fn = lambda theta: [cp.sum_squares(L @ (theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type in ['Filippi']:
            sigma = 1./4.
            V = self.x.T @ self.x / sigma**2 + H

            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            beta = self.confidence_parameter(delta, params, type=type)
            self.set_fn = lambda theta: [cp.sum_squares(L @ (theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type in ["adaptive-AB"]:
            sigma = 1./4.
            V = self.x.T @ self.x / sigma**2 + H
            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            beta = self.confidence_parameter(delta, params, type=type)
            self.set_fn = lambda theta:  [cp.sum_squares(L@(theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type == "LR":
            beta = self.confidence_parameter(delta, params, type=type)
            set = self.lr_confidence_set(theta, beta, params)

        else:
            raise NotImplementedError("The desired confidence set type is not supported.")

        self.set = set
        self.fitted = True

        return set

    def information_matrix(self):
        V = self.x.T@self.x/self.sigma
        return V

    def confidence_parameter(self, delta, params, type = None):
        H = params['regularizer_hessian']
        lam = torch.max(torch.linalg.eigvalsh(H))
        B = params['bound']

        if type is None or type == "none" or type == "laplace":
            # this is a common heuristic
            beta =  2.0
        elif type == "adaptive-AB":
            sigma = 1./4.
            V = self.x.T @ self.x / sigma ** 2 + H
            beta = 2 * np.log(1. / delta) + (torch.logdet(V + H) - torch.logdet(H)) + lam * B

        elif type == "LR":
            # this is based on sequential LR test
            beta = self.confidence_parameter_likelihood_ratio(delta, params)

        elif type == "Faubry":
            H = params['regularizer_hessian']
            lam = H[0., 0]
            beta = np.sqrt(lam)/2.
        else:
            raise NotImplementedError("Not implemented")
        return beta

    def get_torch_objective(self):
        raise NotImplementedError("Implement me please.")
