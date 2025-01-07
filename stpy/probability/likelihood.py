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
        """Evaluate log-probability of a specific value f"""
        pass

    @abstractmethod
    def scale(self, err = None, bound = None):
        """ Return the inverse of strong-convexity parameter of the likelihood"""
        return

    @abstractmethod
    def normalization(self, d):
        """
        Return the normalization constant of the likelihood
        :param d:
        :return:
        """
        return

    @abstractmethod
    def evaluate_datapoint(self, f, d, mask = None):
        """
        Evaluate the likelihood of a specific data point
        :param f: prediction
        :param d: data
        :param mask: weighting of the data
        :return:
        """
        pass

    @abstractmethod
    def get_confidence_set_cvxpy(self, theta, type, params, delta):
        """
        Return the cvxpy set constraint
        :param theta:
        :param type:
        :param params:
        :param delta:
        :return:
        """
        pass

    @abstractmethod
    def information_matrix(self, theta_fit):
        """
        Get an information matrix
        :param theta_fit:
        :return:
        """
        pass


    @abstractmethod
    def get_objective_cvxpy(self, mask = None):
        """
        Return the cvxpy objective of the negative log-likelihood
        :param mask:
        :return:
        """
        pass

    @abstractmethod
    def get_objective_torch(self):
        """
        Return the torch objective of the negative log-likelihood
        :return:
        """
        pass


    def add_data_point(self, d, weight = None):
        x,y = d
        self.x = torch.vstack(self.x,x)
        self.y = torch.vstack(self.y,y)
        if weight is None and self.weights is not None:
            self.weights = torch.vstack(self.weights, torch.ones(weight))
        self.fitted = False

    def load_data(self, D, weights = None):
        self.x, self.y = D
        self.weights = weights
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
        print ("Evidence:", evidence)
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


   



