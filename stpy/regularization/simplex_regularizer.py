from stpy.regularization.regularizer import Regularizer
import cvxpy as cp
import numpy as np
import torch
class ProbabilityRegularizer(Regularizer):

    def __init__(self, lam=1, w=None, d=1, **kwargs):
        super().__init__(lam)
        self.lam = lam
        if w is None:
            self.w = torch.ones(d).double()/d
        else:
            self.w = w
        self.convex = True
        self.dcp = True
        self.d = d
        self.name = "default"
class SupRegularizer(ProbabilityRegularizer):

    def __init__(self, constrained = False, version = '1',**kwargs):
        super().__init__(**kwargs)
        self.convex = False
        self.name = "sup"
        self.constrained = constrained
        self.version = version
    def get_regularizer_cvxpy(self):
        pass

    def get_cvxpy_objectives_constraints_variables(self, d):
        if not self.constrained:
            print (d, self.w )
            objectives = [lambda x: cp.inv_pos(x[i])*self.lam/self.w[i] for i in range(d)]
            constriants = [lambda x: [] for i in range(d)]
            return objectives, constriants, []
        elif self.version == '1':
            objectives = [lambda x: 0. for i in range(d)]
            #constriants = [lambda x: [cp.inv_pos(x[i])<=1/self.lam]+[cp.max(x)<=x[i]] for i in range(d)]
            constriants = [lambda x: [x[i] >= self.lam]  for i in range(d)]
            return objectives, constriants, []
        else:
            objectives = [lambda x: 0.]
            I = np.eye(d)
            constriants = [lambda x: [ I*self.lam*cp.sum(x) << d*cp.diag(x)]]
            return objectives, constriants, []
    def eval(self, theta):
        return self.lam/torch.max(self.w*theta)

class DirichletRegularizer(ProbabilityRegularizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "dirichlet"

    def get_regularizer_cvxpy(self):
        return lambda x: cp.sum((self.w-1)@cp.log(x)) * self.lam

    def eval(self, theta):
        return self.lam / torch.sum(torch.abs(theta))

class WeightedAitchisonRegularizer(ProbabilityRegularizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dcp = False
        self.name = "aitchison"


    def get_regularizer_cvxpy(self):
        def reg(x):
           # outer = sum([cp.log(x[j])*cp.log(x[i]) for i,j in zip(range(self.d),range(self.d)) if i!=j])
            return 2*self.lam*(cp.sum(cp.log(x)**2))

        return reg
    def eval(self, theta):
        return self.lam / torch.sum(torch.abs(theta))


class L1MeasureRegularizer(ProbabilityRegularizer):
    def get_regularizer_cvxpy(self):
        return lambda x: cp.norm1(x)*self.lam

    def eval(self, theta):
        return self.lam/torch.sum(torch.abs(theta))

