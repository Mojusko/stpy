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


    def confidence_parameter_prior_posterior(self, delta,params):
        H = params['regularizer_hessian']
        print ("hessian", H)
        sigma = params['sigma']
        n = self.x.size()[0]
        x = self.x @ torch.sqrt(torch.inverse(H).double())
        K = (x@x.T + (sigma**2)*torch.eye(n))
        evidence_of_the_data = -0.5*self.y.T@torch.linalg.solve(K,self.y)-0.5*torch.linalg.slogdet(K)[1]-(n/2)*np.log(2*np.pi) + (n/2)*np.log(2*np.pi*sigma**2) ## last terms come from the other
        return np.log(1./delta) - evidence_of_the_data


    def confidence_parameter_prior_posterior_weighted(self, delta,params):
        H = params['regularizer_hessian']
        sigma = params['sigma']
        ev = params['evidence']

        print ("Evidence:", ev)

        n = self.x.size()[0]
        d = self.x.size()[1]

        bSigma = torch.diag(1./torch.Tensor(ev).double()) * sigma**2

        K = (self.x @ self.x.T + torch.max(H) * (bSigma))
        y = self.y

        evidence_of_the_data = -0.5*y.T@torch.linalg.solve(K,y)-0.5*torch.linalg.slogdet(2*np.pi*K)[1] ## last terms come from the other
        return np.log(1./delta) - evidence_of_the_data

        #est = estimators[-1]
        #invV = torch.inverse(V)
        # for i in range(len(estimators)-1):
        #     xx = x[i,:].view(1,-1)*np.sqrt(ev[i])
        #     yy = y[i,:].view(1,-1)*np.sqrt(ev[i])
        #     V = (self.x[:i-1,:].T @ self.x[:i-1,:] / (sigma ** 2) + torch.max(H) * torch.eye(d))
        #     invV = torch.inverse(V)
        #     est = estimators[i]
        #     #val = self.evaluate_datapoint(est, (xx, yy), mask=1)
        #     val = (xx@est - yy)**2
        #     print ("i", i, ":",val)
        #     posterior_variance = xx@invV@xx.T + sigma**2
        #     #evidence_of_the_data += ev[i]*(-0.5*np.log(2*np.pi) - 0.5 *xx@invV@xx.T*sigma**2 - val)#-0.5*np.log(2*np.pi*sigma**2))
        #
        #     evidence_of_the_data+=ev[i]*(-0.5*np.log(2*np.pi*posterior_variance) - val/(2*posterior_variance) + 0.5*np.log(2*np.pi*sigma**2))
        #     print ("log, DELTA:", np.log(1/delta))
        #     print ( 0.5*np.log(2*np.pi*sigma**2))
        #     print (0.5*np.log(2*np.pi*posterior_variance))
        #     print (val/(2*posterior_variance))
        #return np.log(1./delta) - evidence_of_the_data

    def prior_posterior_lr_confidence_set_cvxpy_weighted(self, theta, beta, params):
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
        def gauss_likelihood(theta): return cp.sum(cp.multiply(np.array(evidence).reshape(-1,1),cp.square((self.x @ theta - self.y)) / (2 * (sigma ** 2))))
        self.set_fn = lambda theta:  [gauss_likelihood(theta) <= beta + np.sum(np.array(evidence))*(1/2)*np.log(2*np.pi*sigma**2)]
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
        def gauss_likelihood(theta): return cp.sum(cp.square((self.x @ theta - self.y)) / (2 * (sigma ** 2)))
        self.set_fn = lambda theta:  [gauss_likelihood(theta) <= beta]
        set = self.set_fn(theta)
        return set



