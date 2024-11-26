from stpy.regularization.regularizer import Regularizer
from stpy.regularization.constraints import Constraints
import cvxpy as cp
from typing import Union
import torch

class SDPConstraint(Constraints):

    def __init__(self,
                 type="trace",
                 rank=1.,
                 trace_constraint = None,
                 matrix_bound = None,
                 lambda_max_constraint = None,
                 pattern = None,
                 nuclear = None,
                 linear_constraint: None = None,
                 bilinear_constraint = None,
                 regularization = False):

        super().__init__()
        self.bilinear_constraint = bilinear_constraint
        self.linear_constraint = linear_constraint
        self.trace_constraint = trace_constraint
        self.lambda_max_constraint = lambda_max_constraint
        self.psd_constraint = "Yes"
        self.matrix_bound = matrix_bound
        self.type = type
        self.rank = rank
        self.pattern = pattern
        self.custom_regularization= None
        self.nuclear = nuclear
        self.regularization = regularization
        self.fit_params()

    def fit_params(self):
        if self.type == "stable-rank":
            self.matrix_bound = self.rank

    def eval(self, Gamma):
        if self.type == "trace":
            return torch.trace(Gamma)*self.trace_constraint
        elif self.type == "lambda_max":
            return torch.lambda_max(Gamma) * self.lambda_max_constraint
        elif self.type == "stable-rank":
            return torch.norm(Gamma, "nuc")
        elif self.type == "custom":
            return self.custom_regularization(Gamma)
    def get_type(self):
        return self.type

    def get_regularization_cvxpy(self, A, l, s_value):
        if self.regularization:
            return self.linear_constraint.get_regularization_cvxpy(A, l, s_value)

    def get_constraint_cvxpy(self,A,l,s_value):
        constraints = []

        # add a classical psd constraint
        if self.matrix_bound is not None and self.pattern is None:
            constraints+=[cp.trace(A) <= self.matrix_bound * l] + [cp.lambda_max(A) <= l]

            # restrict the max eigenvalue
            if s_value is not None:
                constraints += [l<=s_value]

        # trace regularization
        if self.trace_constraint is not None:
            constraints += [cp.trace(A) <= self.trace_constraint]

        # lambda_max regularization
        if self.lambda_max_constraint is not None:
            constraints += [cp.lambda_max(A) <= self.lambda_max_constraint]

        # lambda_max regularization
        if self.nuclear is not None:
            constraints += [cp.norm(A, "nuc") <= self.nuclear]

        if self.linear_constraint is not None and self.regularization is False:
            constraints += self.linear_constraint.get_constraint_cvxpy(A,l,s_value)


        if self.custom_regularization is not None:
            constraints += [self.custom_regularization(A,l,s_value)]

        return constraints

class SDPLinearInequality():

    def __init__(self, Bs, cs):
        self.Bs = Bs
        self.cs = cs
        self.type = 'linear'
    def get_constraint_cvxpy(self,A,l,s_value):
        constraints = []
        for B,c in zip(self.Bs, self.cs):
            constraints += [cp.trace(B@A) - c <= 0]
        return constraints

    def get_regularization_cvxpy(self,A,l,s_value):
        objective = None
        for B,c in zip(self.Bs, self.cs):
            if objective is None:
                objective = cp.trace(B@A)*c
            else:
                objective += cp.trace(B @ A) * c
        return objective