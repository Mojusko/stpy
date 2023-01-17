import cvxpy as cp
import torch

from stpy.probability.likelihood import Likelihood


class GaussianLikelihood(Likelihood):

    def __init__(self, sigma = 0.1, Sigma=None):
        super().__init__()

        self.sigma = sigma
        self.Sigma = Sigma

    def evaluate_log(self, f):
        if self.Sigma is None:
            res = torch.sum((f - self.y)**2)/self.sigma**2
        else:
            res = ((f - self.y).T @ torch.inverse(self.Sigma.T@self.Sigma)  @ (f - self.y) )
        return res

    def load_data(self, D):
        self.x, self.y = D

    def add_data_point(self, d):
        x,y = d
        self.x = torch.vstack(self.x,x)
        self.y = torch.vstack(self.y,y)

    def get_torch_objective(self):
        pass

    def evaluate(self, theta):
        pass

    def get_cvxpy_objective(self):
        if self.Sigma is None:
            def likelihood(theta): return cp.sum_squares(self.x@theta - self.y)/self.sigma**2

        else:
            def likelihood(theta): return cp.matrix_frac(self.x@theta - self.y,self.Sigma.T@self.Sigma)
        return likelihood

    def information_matrix(self):
        if self.Sigma is None:
            V = self.x.T@self.x/self.sigma**2
        else:
            V = self.x.T@self.Sigma.T@self.Sigma@self.x
        return V

    def get_cvxpy_confidence_set(self, theta, type = None, information = [], delta = 0.1):

        if type is None:
            theta_fit = information[0]
            H = information[1]
            beta = 2.
            V = self.information_matrix() + H
            set = [cp.quad_form(theta - theta_fit, V) <= beta]
            return set

        elif type == "LR_static":
            theta_fit = information[0]
            beta = 2.
            V = self.information_matrix()*self.sigma**2 + 10e-8 * torch.eye(n = theta_fit.size()[0]).double()
            set = [cp.quad_form(theta - theta_fit, V) <= beta*self.sigma**2]
            return set


    def confidence_parameter(self, type = None):
        ## TODO: Add here theoretical for adaptive and fixed designs
        if type is None:
            return 2.0
