from stpy.regularization.regularizer import Regularizer
from stpy.regularization.regularizer import L2Regularizer
from typing import List
import torch
import numpy as np

class GroupRegularizer(Regularizer):

    def __init__(self,
                 indices : List[List[int]],
                 dim : int,
                 active = 1,
                 base_regularizer : Regularizer = L2Regularizer(lam = 1.)):
        self.indices = indices
        self.active = active
        self.dim = dim
        self.convex = False
        self.base_regularizer = base_regularizer
        self.discrete = True

    def get_regularizer_cvxpy(self, mask = None):
        pass

    def get_list_regularizer_cvxpy(self):
        masks = self.get_masks()
        regs = []
        for mask in masks:
            regs.append(self.base_regularizer.get_regularizer_cvxpy())
        return regs

    def eval(self, theta):
        masks = self.get_masks()
        count = 0
        for mask in masks:
            if (theta[mask])**2> 1e-10:
                count +=1
        if count > self.active:
            return np.inf
        else:
            return 0.


    def get_masks(self):
        masks = []
        for ind in self.indices:
            mask = torch.zeros(self.dim, dtype = torch.bool)
            mask[ind] = True
            masks.append(mask)
        return masks

    def is_convex(self):
            return self.convex

    def hessian(self, theta_fit):
        return self.base_regularizer.hessian(theta_fit)