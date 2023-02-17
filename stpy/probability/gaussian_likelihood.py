import cvxpy as cp
import numpy as np
import torch
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood
import scipy

class GaussianLikelihood(Likelihood):

    def __init__(self, sigma = 0.1, Sigma=None):
        super().__init__()

        self.sigma = sigma
        self.Sigma = Sigma

    def scale(self):
        if self.Sigma is None:
            return self.sigma**2
        else:
            return torch.max(self.Sigma.T@self.Sigma)

    def evaluate_log(self, f):
        if self.Sigma is None:
            res = torch.sum((f - self.y)**2)/self.sigma**2
        else:
            res = ((f - self.y).T @ torch.inverse(self.Sigma.T@self.Sigma)  @ (f - self.y) )
        return res

    def load_data(self, D):
        self.x, self.y = D
        self.fitted = False

    def add_data_point(self, d):
        x,y = d
        self.x = torch.vstack(self.x,x)
        self.y = torch.vstack(self.y,y)
        self.fitted = False

    def evaluate_point(self, theta, d):
        x,y = d
        if self.Sigma is None:
            return ((x @ theta - y) ** 2)/ self.sigma ** 2
        else:
            return (x @ theta - y).T @ torch.linalg.inv(self.Sigma.T @ self.Sigma) @ (
                            x @ theta - y)

    def get_torch_objective(self):

        if self.Sigma is None:
            def likelihood(theta): return torch.sum((self.x@theta - self.y)**2)/(self.sigma**2)

        else:
            def likelihood(theta): return (self.x@theta - self.y).T@torch.linalg.inv(self.Sigma.T@self.Sigma)@(self.x@theta - self.y)
        return likelihood

    def get_cvxpy_objective(self, mask = None):
        if mask is None:
            if self.Sigma is None:
                def likelihood(theta): return cp.sum_squares(self.x@theta - self.y)/(self.sigma**2)

            else:
                def likelihood(theta): return cp.matrix_frac(self.x@theta - self.y,self.Sigma.T@self.Sigma)
        else:
            if self.Sigma is None:
                def likelihood(theta):
                    if torch.sum(mask.int())>0:
                        return cp.sum_squares(self.x[mask,:] @ theta - self.y[mask, :]) / (self.sigma ** 2)
                    else:
                        return cp.sum(theta*0)

            else:
                def likelihood(theta):
                    if torch.sum(mask.int())>0:
                        return cp.matrix_frac(self.x[mask, :] @ theta - self.y[mask, :], self.Sigma.T @ self.Sigma)
                    else:
                        return cp.sum(theta*0)
        return likelihood

    def information_matrix(self, mask = None):
        if mask is None:
            if self.Sigma is None:
                V = self.x.T@self.x/(self.sigma**2)
            else:
                V = self.x.T@self.Sigma.T@self.Sigma@self.x
            return V
        else:
            if self.Sigma is None:
                V = self.x[mask,:].T@self.x[mask,:]/(self.sigma**2)
            else:
                V = self.x[mask,:].T@self.Sigma.T@self.Sigma@self.x[mask,:]
            return V

    def get_cvxpy_confidence_set(self,
                                 theta: cp.Variable,
                                 type: Union[str,None] = None,
                                 params: Dict = {},
                                 delta: float  = 0.1):
        if self.fitted == True:
            return self.set_fn(theta)

        theta_fit = params['estimate']
        H = params['regularizer_hessian']

        if H is not None:
            V = self.information_matrix() + H
        else:
            V = self.information_matrix()

        if type in ["none", None, "fixed"]:
            L = torch.linalg.cholesky(V).double()
            beta = self.confidence_parameter(delta, params, type=type)
            self.set_fn = lambda theta:  [cp.sum_squares(L@(theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type in ["adaptive-AB"]:
            #L = torch.linalg.cholesky(V).double()
            L = torch.from_numpy(scipy.linalg.sqrtm(V.numpy()))
            beta = self.confidence_parameter(delta, params, type=type)
            self.set_fn = lambda theta:  [cp.sum_squares(L@(theta - theta_fit)) <= beta]
            set = self.set_fn(theta)

        elif type in ["adaptive-optimized"]:
            beta = self.confidence_parameter(delta, params, type=type)
            sqrtV = scipy.linalg.sqrtm(V)
            L = torch.linalg.cholesky(V+sqrtV).double()
            self.set_fn = lambda theta: [cp.sum_squares(L @ (theta - theta_fit)) <= beta]

        elif type == "LR":
            beta = self.confidence_parameter_likelihood_ratio(delta, params)
            set = self.lr_confidence_set(theta, beta, params)

        else:
            raise NotImplementedError("The desired confidence set type is not supported.")

        self.set = set
        self.fitted = True

        return set

    def confidence_parameter(self, delta, params, type = None):
        if type is None or type == "none":
            # this is a common heuristic
            beta =  2.0
        elif type == "LR" or type == "LR-vovk":
            # this is based on sequential LR test
            beta = self.confidence_parameter_likelihood_ratio(delta, params)

        else:
            if 'd_eff' in params.keys():
                n = self.x.size()[0]
                d = params['d_eff']
            else:
                d = params['m']

            B = params['bound']
            H = params['regularizer_hessian']
            lam = torch.max(torch.linalg.eigvalsh(H))

            if type == "fixed":
                # this is fixed design
                beta = d + 2 * np.log(1 / delta) + 2 * np.sqrt(d * np.log(1 / delta)) + lam*B

            elif type == "adaptive-AB":
                # this takes the pseudo-maximization with a fixed mixture
                V = self.information_matrix()
                beta = 2*np.log(1./delta) + (torch.logdet(V+H) - torch.logdet(H)) + lam*B

            elif type == "adaptive-optimized":
                # this takes at each step the pseudo-maximization the past value
                kernel_object = params['kernel_object']
                #K = kernel_object.kernel(self.x, self.x)
                #Kold = kernel_object.kernel(self.x[0:-1], self.x[0:-1])
                Info = self.information_matrix()
                sqrtV = scipy.linalg.sqrtm(I)
                V = Info + H
                mask = torch.ones(size = (self.x.size()[0],1)).view().bool()
                mask[-1] = False
                Vold = self.information_matrix(mask=mask)
                beta = 2 * np.log((1. / delta)) + (torch.slogdet(V+Vold+H)[1] - torch.slogdet(H+Vold)[1]) + lam * B
            else:
                raise NotImplementedError("The desired confidence set type is not supported.")

        return beta