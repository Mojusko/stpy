import cvxpy as cp
import numpy as np
import torch
from typing import Union, Dict, List
from stpy.probability.likelihood import Likelihood
from stpy.probability.gaussian_likelihood import GaussianLikelihood
import scipy



class MultinomialLikelihood(Likelihood):

    def __init__(self):
        super().__init__()
        
    def evaluate_datapoint(self, theta, d, mask=None):
        if mask is None:
            mask = 1.
        x, y = d
        assert x.dim() == 2, f"Expected x to be (K, feature_dim), got shape {x.shape}"
        assert y.dim() == 1, f"Expected y to be (K,), got shape {y.shape}"
        assert x.size(0) == y.size(0), f"K dimension mismatch: {x.size(0)} vs {y.size(0)}"
        
        logits = x @ theta
        max_logit = torch.max(logits)
        log_probs = logits - max_logit - torch.log(torch.sum(torch.exp(logits - max_logit)))
        return -torch.sum(y * log_probs) * mask

    def get_objective_cvxpy(self, mask=None):
        if mask is None:
            def likelihood(theta):
                x_reshaped = self.x.reshape(-1, self.x.shape[-1])
                logits = x_reshaped @ theta
                logits = logits.reshape((self.x.shape[0], self.x.shape[1]), order='C')
                return cp.sum(cp.log_sum_exp(logits, axis=1) - cp.sum(cp.multiply(self.y, logits), axis=1))
        else:
            def likelihood(theta):
                x_reshaped = self.x.reshape(-1, self.x.shape[-1])
                logits = x_reshaped @ theta
                logits = logits.reshape((self.x.shape[0], self.x.shape[1]), order='F')
                return cp.sum(mask * (cp.log_sum_exp(logits, axis=1) - cp.multiply(self.y, logits)))
        return likelihood

    def load_data(self, D, weights=None):
        self.x, self.y = D
        assert self.x.dim() == 3, "Expected x to be (n_groups, K, feature_dim)"
        assert self.y.dim() == 2, "Expected y to be (n_groups, K)"
        assert self.x.size(0) == self.y.size(0), "Number of groups must match"
        assert self.x.size(1) == self.y.size(1), "K dimension must match"
        
        self.weights = weights
        self.fitted = False

    def evaluate_log(self, f):
        raise NotImplementedError("evaluate_log not implemented")

    def scale(self, err=None, bound=None):
        raise NotImplementedError("scale not implemented")

    def normalization(self, d):
        raise NotImplementedError("normalization not implemented")

    def information_matrix(self, theta_fit):
        raise NotImplementedError("information_matrix not implemented")

    def get_confidence_set_cvxpy(self, theta, type, params, delta):
        raise NotImplementedError("get_confidence_set_cvxpy not implemented")

    def get_objective_torch(self):
        raise NotImplementedError("get_objective_torch not implemented")
