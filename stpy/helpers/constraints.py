from abc import ABC, abstractmethod
from typing import Callable
from stpy.helpers.helper import cartesian
import cvxpy as cp
import numpy as np
import torch
import copy


class Constraints(ABC):
    def __init__(self):
        self.convex = True
        pass

    @abstractmethod
    def get_cvxpy_constraint(self, theta):
        pass


class CustomConstraint(Constraints):

    def __init__(self, custom_function: Callable, custom_function_cvxpy: Callable):
        super().__init__()
        self.fn = custom_function
        self.fn_cvxpy = custom_function_cvxpy

    def get_cvxpy_constraint(self, theta):
        return self.fn_cvxpy(theta)

class LinearEqualityConstraint(Constraints):

    def __init__(self):
        pass

class LinearInequalityConstraint(Constraints):

    def __init__(self):
        pass

class AbsoluteValueConstraint(Constraints):

    def __init__(self,c = None):
        if c is None:
            self.c = 1.
        else:
            self.c = c
    def get_cvxpy_constraint(self, theta):
        set = [cp.norm1(theta) <= self.c]
        return set

class QuadraticInequalityConstraint(Constraints):
    """
    xQx - b@x <= c
    """
    def __init__(self,Q, b = None, c = None ):
        self.Q = Q
        if c is None:
            self.c = 1.
        else:
            self.c = c
        if b is None:
            self.b = torch.zeros(size = (Q.size()[0],1))
        else:
            self.b = b
    def get_cvxpy_constraint(self,theta):
        set = [cp.quad_form(theta, self.Q) - self.b.T@theta <= self.c]
        return set


class NonConvexNormConstraint(Constraints):

    def __init__(self, q, c, d):
        super().__init__()
        self.q = q
        self.c = c
        self.d = d
        self.convex = False
        self.construct(q,d)

    def construct(self,q,d):
        self.vertex_description = []
        square = cartesian([[-q,q] for _ in range(d)])
        for i in range(2*d):
            polytope = copy.copy(square)
            zero = np.zeros(d).reshape(1,-1)
            appex = copy.copy(zero)
            appex[0,i//2] = (float(i % 2)-0.5)*2.
            polytope = np.concatenate((appex,polytope))
            self.vertex_description.append(polytope)
        print (self.vertex_description)

    def get_cvxpy_constraint(self, theta):
        pass

    def get_list_cvxpy_constraints(self, theta):
        pass

    def eval(self,theta):
        out = []
        for i in range(self.d):
            out.append(self.A[i] @ theta <= self.b[i])
        return max(out)

if __name__ == "__main__":
    c = NonConvexNormConstraint(0.5,1,2)

