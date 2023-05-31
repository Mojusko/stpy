from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np
import torch

class Likelihood(ABC):

    def __init__(self):
        self.fitted = False
        pass

    @abstractmethod
    def evaluate_log(self, f):
        pass

    @abstractmethod
    def scale(self, err = None, bound = None):
        return

    @abstractmethod
    def normalization(self, d):
        return

    @abstractmethod
    def evaluate_datapoint(self, f, d, mask = None):
        pass

    @abstractmethod
    def get_confidence_set_cvxpy(self, theta, type, params, delta):
        pass

    @abstractmethod
    def information_matrix(self, theta_fit):
        pass


    @abstractmethod
    def get_objective_cvxpy(self, mask = None):
        pass

    @abstractmethod
    def get_objective_torch(self):
        pass


    def add_data_point(self, d):
        x,y = d
        self.x = torch.vstack(self.x,x)
        self.y = torch.vstack(self.y,y)
        self.fitted = False

    def load_data(self, D):
        self.x, self.y = D
        self.fitted = False

    def confidence_parameter_likelihood_ratio(self, delta, params):
        """
        Evaluates point and weight appropriately in the running likelihood ratio test
        :param delta:
        :param params:
        :return:
        """
        evidence = params['evidence']
        estimators = params['estimator_sequence']

        val = 0.
        for i in range(len(estimators)-1):
            ev = evidence[i]
            est = estimators[i]
            if est is not None:
                xx = self.x[i,:].view(1,-1)
                yy = self.y[i,:].view(1,-1)
                val += self.evaluate_datapoint(est, (xx, yy), mask = ev)
        val = np.log(1/delta) + val
        return val

    def lr_confidence_set_cvxpy(self, theta, beta, params):
        """
        Return the cvxpy set constraint
        :param theta:
        :param beta:
        :param params:
        :return:
        """
        evidence = torch.Tensor(params['evidence']).bool()
        self.set_fn = lambda theta:  [self.get_objective_cvxpy(mask = evidence)(theta) <= beta]
        set = self.set_fn(theta)
        return set


    def confidence_parameter_prior_posterior(self, delta,params):
        H = params['regularizer_hessian']
        sigma = params['sigma']
        n = self.x.size()[0]
        K = (self.x@self.x.T + torch.max(H)*sigma**2*torch.eye(n))
        evidence_of_the_data = -0.5*self.y.T@torch.linalg.solve(K,self.y)-0.5*torch.linalg.slogdet(K)[1]#-(n/2)*np.log(2*np.pi) ## remove this as in likelihood not added
        evidence_of_the_data = evidence_of_the_data #- np.log(2*np.pi*sigma**2)
        return np.log(1./delta) - evidence_of_the_data

    def prior_posterior_lr_confidence_set_cvxpy(self, theta, beta, params):
        """
        Return the cvxpy set constraint
        :param theta:
        :param beta:
        :param params:
        :return:
        """
        # create a Gaussian likelihood
        sigma = params['sigma']
        def gauss_likelihood(theta): return cp.sum_squares(self.x @ theta - self.y) / (2 * sigma ** 2)
        self.set_fn = lambda theta:  [gauss_likelihood(theta)<= beta]
        set = self.set_fn(theta)
        return set



