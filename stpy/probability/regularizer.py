from abc import ABC, abstractmethod

import cvxpy as cp
import torch

from stpy.helpers.constraints import CustomConstraint


class Regularizer(ABC):

    def __init__(self, lam=1.):
        self.lam = lam
        self.groups = None

    @abstractmethod
    def eval(self, theta):
        pass

    def is_convex(self):
        return True

    @abstractmethod
    def get_cvxpy_regularizer(self):
        def reg(theta): return 0
        return reg

    def get_cvxpy_constraint_set(self, theta, c):
        return [self.get_cvxpy_regularizer()(theta) <= c]

    def get_constraint_object(self,c):
        return CustomConstraint(None,lambda theta: self.get_cvxpy_constraint_set(theta,c))

class L2Regularizer(Regularizer):

    def __init__(self, lam=1.):
        super().__init__(lam = lam)

    def get_cvxpy_regularizer(self):
        def reg(theta): return self.lam*cp.sum_squares(theta)
        return reg

    def eval(self, theta):
        return self.lam*torch.sum(theta**2)

    def hessian(self, theta):
        return self.lam * torch.eye(n = theta.size()[0]).double()

class NonConvexLqRegularizer(Regularizer):

    def __init__(self, lam=1., q = 0.5):
        super().__init__(lam = lam)
        self.q = q

    def eval(self, theta):
        return self.lam*torch.sum(torch.abs(theta)**self.q)

    def hessian(self, theta):
        return None

    def is_convex(self):
        return False

    def get_cvxpy_regularizer(self, eta):
        def reg(theta):
            norm = cp.sum_squares(theta/eta.reshape(-1,1))
            return self.q*0.5*norm*self.lam
        return reg

class GroupNonCovexLqRegularizer(NonConvexLqRegularizer):

    def __init__(self, lam=1., q = 0.5, groups = None):
        super().__init__(lam = lam)
        self.q = q
        self.groups = groups

    def eval(self, theta):
        val = None
        for group in self.groups:
            if val is None:
                val = torch.norm(theta[group])**self.q
            else:
                val += torch.norm(theta[group]) ** self.q
        return self.lam*val

    def get_cvxpy_regularizer(self, eta):
        def reg(theta):
            val = None
            for i,group in enumerate(self.groups):
                if val is None:
                    val = cp.sum_squares(theta[group])/eta[i].reshape(-1,1)
                else:
                    val += cp.sum_squares(theta[group])/eta[i].reshape(-1,1)
            return val*self.lam
        return reg


class L1Regularizer(Regularizer):
    def __init__(self, lam=1.):
        super().__init__(lam = lam)

    def get_cvxpy_regularizer(self):
        def reg(theta):
            return self.lam*cp.norm1(theta)
        return reg

    def eval(self, theta):
        return self.lam*torch.sum(torch.abs(theta))

    def hessian(self, theta):
        return None



class GroupL1L2Regularizer(Regularizer):

    def __init__(self, lam = 1., groups = None):
        self.groups = groups
        self.lam = lam
        pass

    def eval(self, theta):
        norm = 0
        for group in self.groups:
            norm += torch.linalg.norm(theta[group])
        return norm**2 * self.lam

    def get_cvxpy_regularizer(self):
        def reg(theta):
            norm = None
            for group in self.groups:
                if norm is None:
                    norm = cp.norm2(theta[group])
                else:
                    norm += cp.norm2(theta[group])
            return cp.square(norm)*self.lam
        return reg

    def hessian(self, theta):
        return None

class NestedGroupL1Regularizer(Regularizer):
    def __init__(self, lam = 1., groups = None, weights = None):
        self.groups = groups
        self.lam = lam
        self.weights = weights
        pass

    def eval(self, theta):
        norm = 0
        for i, group in enumerate(self.groups):
            norm += self.weights[i]*torch.sum(torch.abs(theta[group]))
        return norm**2 * self.lam

    def get_cvxpy_regularizer(self):

        def reg(theta):
            norm = None
            for i, group in enumerate(self.groups):

                if norm is None:
                    norm = self.weights[i] * cp.norm1(theta[group])
                else:
                    norm += self.weights[i] * cp.norm1(theta[group])

            return norm*self.lam

        return reg

    def hessian(self, theta):
        return None

class NestedGroupL1L2Regularizer(Regularizer):

    def __init__(self, lam = 1., groups = None, weights = None):
        self.groups = groups
        self.lam = lam
        self.weights = weights
        pass

    def eval(self, theta):
        norm = 0
        for i, group in enumerate(self.groups):
            norm += self.weights[i] * torch.linalg.norm(theta[group])
        return norm**2 * self.lam

    def get_cvxpy_regularizer(self):

        def reg(theta):
            norm = None
            for i, group in enumerate(self.groups):

                if norm is None:
                    norm = self.weights[i] * cp.norm2(theta[group])
                else:
                    norm += self.weights[i] * cp.norm2(theta[group])

            return cp.square(norm)*self.lam

        return reg

    def hessian(self, theta):
        return None