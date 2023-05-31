import cvxpy as cp
import numpy as np
import torch
import scipy
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood
from stpy.probability.gaussian_likelihood import GaussianLikelihood

class LaplaceLikelihood(GaussianLikelihood):

    def __init__(self, b = 0.1):
        super().__init__()
        self.b = b

    def scale(self, err = None, bound = None):
        return self.b

    def evaluate_log(self, f):
        res = torch.sum(torch.abs(f - self.y))/self.b
        return res

    def evaluate_datapoint(self, theta, d, mask = None):
        if mask is None:
            mask = 1.
        x, y = d
        return mask* (torch.abs(x @ theta - y)) / self.b

    def get_objective_cvxpy(self, mask = None):
        if mask is None:
             def likelihood(theta): return cp.sum(cp.abs(self.x@theta - self.y)/self.b)
        else:
            def likelihood(theta):
                if torch.sum(mask.int())>0:
                    return cp.sum(cp.abs(self.x[mask,:]@theta - self.y[mask,:])/self.b)
                else:
                    return cp.sum(theta*0)
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

        if H is not None:
            V = self.information_matrix() + H
        else:
            V = self.information_matrix()

        if type in ["none","sub-exp"]:
            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            beta = self.confidence_parameter(delta, params, type=type)
            self.set_fn = lambda theta: [cp.sum_squares(L @ (theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type == "adaptive-AB":
            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            beta = self.confidence_parameter(delta, params, type=type)
            self.set_fn = lambda theta: [cp.sum_squares(L @ (theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type == "LR":
            beta = self.confidence_parameter_likelihood_ratio(delta, params)
            set = self.lr_confidence_set_cvxpy(theta, beta, params)

        else:
            raise NotImplementedError("The desired confidence set type is not supported.")
        print(type, "USING BETA: ", beta)

        self.set = set
        self.fitted = True

        return set

    def information_matrix(self):
        V = self.x.T@self.x/(2*self.b)**2
        return V


    def get_objective_torch(self, mask = None):
        if mask is None:
            def likelihood(theta): return torch.sum(torch.abs(self.x@theta - self.y)/self.sigma)
        else:
            def likelihood(theta):
                if torch.sum(mask.int())>0:
                    return torch.sum(torch.abs(self.x[mask,:]@theta - self.y[mask,:])/self.sigma)
                else:
                    return torch.sum(theta*0)
        return likelihood



    def confidence_parameter(self, delta, params, type = None):
        print (type)

        if type is None or type == "none":
            beta =  2.0 * np.log(1/delta)

        else:
            if 'd_eff' in params.keys():
                n = self.x.size()[0]
                d = params['d_eff']
            else:
                d = params['m']

            B = params['bound']
            H = params['regularizer_hessian']
            lam = torch.max(torch.linalg.eigvalsh(H))

            if type == "sub-exp":
                # this takes the pseudo-maximization with a fixed mixture
                V = self.information_matrix()
                L = 1.
                size = V.size()[0]
                beta = (lam*(B + self.b/L) + L/(self.b*np.sqrt(lam))*(d*np.log(2)+np.log(1./delta)+0.5*torch.slogdet(V*lam+torch.eye(size))[1]))
            elif type == "adaptive-AB":
                V = self.information_matrix()
                beta = 2*np.log(1./delta) + (torch.logdet(V+H) - torch.logdet(H)) + lam*B
            else:
                raise NotImplementedError("given confidence sets are not implemented.")
        return beta