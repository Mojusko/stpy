from stpy.regularization.regularizer import Regularizer
from stpy.regularization.constraints import Constraints
import cvxpy as cp

class SDPConstraint(Constraints):

    def __init__(self, type="trace", rank=1.):

        super().__init__()

        self.trace_constraint = None
        self.lambda_max_constraint = None
        self.psd_constraint = "Yes"
        self.matrix_bound = 1.
        self.type = type
        self.rank = rank
        self.custom_regularization= None

        self.fit_params()
    def fit_params(self):
        if self.type == "stable-rank":
            self.matrix_bound = self.rank

    def get_type(self):
        return self.type

    def get_constraint_cvxpy(self,A,l,s_value):
        constraints = []

        # add a classical psd constraint
        if self.matrix_bound is not None:
            constraints+=[cp.trace(A) <= self.matrix_bound * l] + [cp.lambda_max(A) <= l]

        # trace regularization
        if self.trace_constraint is not None:
            constraints += [cp.trace(A) <= self.trace_constraint]

        # restrict the max eigenvalue
        if s_value is not None:
            constraints += [l<=s_value]

        # lambda_max regularization
        if self.lambda_max_constraint is not None:
            constraints += [cp.lambda_max(A) <= self.lambda_max_constraint]

        if self.custom_regularization is not None:
            constraints += [self.custom_regularization(A,l,s_value)]

        return constraints