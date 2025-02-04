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
        print ("Evidence:", evidence, "Estimators", len(estimators)-1, "Evidence no.:", len(evidence))
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
        evidence = params['evidence']
        print ("Evidence, in set construction:", evidence)

        self.set_fn = lambda theta:  [self.get_objective_cvxpy()(theta) <= beta]
        set = self.set_fn(theta)

        return set

    def confidence_parameter_prior_posterior(self, delta,params):
        if params['discrete_reg']:
            H = params['regularizer_hessian']
            sigma = params['sigma']

            reg = params['regularizer']
            masks = reg.get_masks()
            total_evidence = 0
            y = self.y

            for mask in masks:
                x = self.x[:,mask]
                Hsel = H[:,mask][mask,:]

                S_t = torch.sum(torch.multiply(x.T, y.view(-1)).T / (sigma ** 2), dim=0).view(-1, 1)
                c = -torch.sum(y ** 2) / (2 * sigma ** 2)
                V_t_lambda = (x.T @ x) / (sigma ** 2) + Hsel
                invV = torch.inverse(V_t_lambda)

                evidence_of_the_data = c + 0.5 * S_t.T @ invV @ S_t + 0.5 * (
                            torch.linalg.slogdet(V_t_lambda)[1] - torch.linalg.slogdet(Hsel)[1])

                total_evidence += evidence_of_the_data/len(masks)

                print ("DISCRETE EVIDENCE", total_evidence)
            return np.log(1. / delta) - total_evidence

        else:
            H = params['regularizer_hessian']
            sigma = params['sigma']

            n = self.x.size()[0]
            #x = self.x @ torch.sqrt(torch.inverse(H).double())
            y = self.y
            x = self.x

            S_t = torch.sum(torch.multiply(x.T,y.view(-1)).T/(sigma**2), dim = 0).view(-1,1)
            c = -torch.sum(y**2)/(2*sigma**2)
            V_t_lambda = (x.T @ x)/(sigma**2) + H
            invV = torch.inverse(V_t_lambda)

            evidence_of_the_data = c  + 0.5 * S_t.T @  invV @ S_t - 0.5* (torch.linalg.slogdet(V_t_lambda)[1] -torch.linalg.slogdet(H)[1])

            print ("EVIDENCE", evidence_of_the_data)
            return np.log(1./delta)-evidence_of_the_data

    #
    # def confidence_parameter_prior_posterior_weighted(self, delta,params):
    #     H = params['regularizer_hessian']
    #     sigma = params['sigma']
    #     ev = params['evidence']
    #
    #     print ("Evidence:", ev)
    #
    #     n = self.x.size()[0]
    #     d = self.x.size()[1]
    #
    #     bSigma = torch.diag(1./torch.Tensor(ev).double()) * sigma**2
    #
    #     K = (self.x @ self.x.T + torch.max(H) * (bSigma))
    #     y = self.y
    #
    #     evidence_of_the_data = -0.5*y.T@torch.linalg.solve(K,y)-0.5*torch.linalg.slogdet(2*np.pi*K)[1] ## last terms come from the other
    #     return np.log(1./delta) + evidence_of_the_data
    # def prior_posterior_lr_confidence_set_cvxpy_weighted(self, theta, beta, params):
    #     """
    #     Return the cvxpy set constraint
    #     :param theta:
    #     :param beta:
    #     :param params:
    #     :return:
    #     """
    #     # create a Gaussian likelihood
    #     sigma = params['sigma']
    #     evidence = params['evidence']
    #     print ("Evidence, in set construction:", evidence)
    #     def gauss_likelihood(theta): return cp.sum(cp.multiply(np.array(evidence).reshape(-1,1),cp.square((self.x @ theta - self.y)) / (2 * (sigma ** 2))))
    #     self.set_fn = lambda theta:  [gauss_likelihood(theta) <= beta + np.sum(np.array(evidence))*(1/2)*np.log(2*np.pi*sigma**2)]
    #     set = self.set_fn(theta)
    #     return set




