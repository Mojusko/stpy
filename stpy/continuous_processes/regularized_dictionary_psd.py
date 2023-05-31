import torch
from typing import Union
import cvxpy as cp
from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.embedding import Embedding
from stpy.probability.likelihood import Likelihood
from stpy.optim.custom_optimizers import bisection
from stpy.regularization.sdp_constraint import SDPConstraint

class RegularizedDictionaryPSD(RegularizedDictionary):

    def __init__(self,
                 embedding: Embedding,
                 likelihood: Likelihood,
                 sdp_constraint: Union[SDPConstraint, None] = None,
                 pattern = 'full',
                 **kwargs):
        super().__init__(embedding, likelihood, **kwargs)

        self.sdp_regularizer = sdp_constraint
        self.pattern = pattern
        self.tightness = 1e-3

    def calculate(self):
        if self.sdp_regularizer.get_type() == "stable-rank":
            # first output is l_value, eigenvalue
            fun = lambda s: -(-self.tightness + self.calculate_simple(s_value=s)[1] - self.calculate_simple(s_value=s)[2])
            res = bisection(fun, 10e-5, 10e5, 30)
            self.calculate_simple(s_value=res)
        else:
            self.calculate_simple(s_value=None)

    def calculate_enumerate(self):
        assert (self.pattern == "diagonal")
        theta = cp.Variable((self.m, 1))
        l = cp.Variable((1,1))

        vals = []
        for i in range(self.m):
            a = cp.Variable((self.m, 1))

            likelihood = self.likelihood.get_objective_cvxpy()
            objective = likelihood(theta)

            if self.regularizer is not None:
                regularizer = self.regularizer.get_regularizer_cvxpy()
                objective += regularizer(theta)
            I = np.eye(self.m)
            A = cp.diag(a)
            constraints = [cp.matrix_frac(theta, A) <= 1, cp.max(a)<=a[i], a>=0, cp.trace(A) <= a[i]]
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver = cp.MOSEK)
            vals.append(prob.value)

        j = np.argmin(vals)
        constraints = [cp.matrix_frac(theta, A) <= 1, cp.max(a) <= a[j], a >= 0, cp.trace(A) <= a[j]]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.MOSEK)
        self.A_fit = torch.from_numpy(A.value )
        self.theta_fit = torch.from_numpy(theta.value)
        #self.l_fit = torch.from_numpy(l.value)

        self.fitted = True
        return prob.value, None, torch.max(torch.linalg.eigvalsh(torch.from_numpy(A.value)))

    def calculate_simple(self, s_value = None):
        theta = cp.Variable((self.m, 1))
        l = cp.Variable((1, 1))

        if self.pattern=="full":
            A = cp.Variable((self.m, self.m))

        elif self.pattern=="diagonal":
            a = cp.Variable(self.m)
            A = cp.diag(a)

        elif self.pattern=="block-diagonal":
            pass

        else:
            raise NotImplementedError("This pattern for PSD estimator is not implemented.")

        likelihood = self.likelihood.get_objective_cvxpy()
        objective = likelihood(theta)

        if self.regularizer is not None:
            regularizer = self.regularizer.get_regularizer_cvxpy()
            objective += regularizer(theta)

        constraints = []
        if self.constraints is not None and self.use_constraint:
            set = self.constraints.get_constraint_cvxpy(theta)
            constraints += set

        constraints = [cp.matrix_frac(theta, A) <= 1, A >> 0]
        constraints += self.sdp_regularizer.get_constraint_cvxpy(A, l, s_value)
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver = cp.MOSEK)
        self.A_fit = torch.from_numpy(A.value )
        self.theta_fit = torch.from_numpy(theta.value)
        self.l_fit = torch.from_numpy(l.value)
        self.fitted = True

        return prob.value, torch.from_numpy(l.value), torch.max(torch.linalg.eigvalsh(torch.from_numpy(A.value)))

    def lcb(self, x: torch.Tensor, sign: float = 1.):
        pass


if __name__ == "__main__":

    import cvxpy as cp
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, ConcatEmbedding, MaskedEmbedding
    from stpy.kernels import KernelFunction
    from stpy.helpers.helper import interval, interval_torch
    from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
    from stpy.continuous_processes.nystrom_fea import NystromFeatures
    from stpy.probability.gaussian_likelihood import GaussianLikelihood

    N = 10
    n = 256
    d = 1
    eps = 0.01
    s = 0.01
    m = 32

    f = lambda x: torch.sin(x * 20) * (x > 0).double()
    Xtrain = interval_torch(n=N, d=1)
    ytrain = f(Xtrain)

    xtest = torch.from_numpy(interval(n, d, L_infinity_ball=1))
    kernel_object = KernelFunction(gamma=0.05, d=1)

    # embedding = HermiteEmbedding(m=m, gamma = 1.)

    embedding1 = NystromFeatures(kernel_object=kernel_object, m=m // 2)
    embedding1.fit_gp(xtest / 2 - 0.5, None)
    embedding2 = NystromFeatures(kernel_object=kernel_object, m=m // 2)
    embedding2.fit_gp(xtest / 2 + 0.5, None)
    embedding = ConcatEmbedding([embedding1, embedding2])

    likelihood = GaussianLikelihood(sigma=s)
    regularization = SDPConstraint(type = "stable-rank")

    estimator = RegularizedDictionaryPSD(embedding, likelihood, sdp_constraint=regularization)
    estimator.load_data((Xtrain,ytrain))
    estimator.fit()

    estimator_diag = RegularizedDictionaryPSD(embedding, likelihood, sdp_constraint=regularization, pattern='diagonal')
    estimator_diag.load_data((Xtrain,ytrain))
    estimator_diag.fit()

    estimator_sup = RegularizedDictionaryPSD(embedding, likelihood, sdp_constraint=regularization, pattern='diagonal')
    estimator_sup.load_data((Xtrain, ytrain))
    estimator_sup.fit()
    estimator_sup.calculate_enumerate()

    print (estimator.l_fit)
    print (torch.max(torch.linalg.eigvalsh(estimator.A_fit)))

    print (estimator_diag.l_fit)
    print (torch.max(torch.linalg.eigvalsh(estimator_diag.A_fit)))


    mu = estimator.mean(xtest)
    mu_diag = estimator_diag.mean(xtest)
    mu_sup = estimator_sup.mean(xtest)

    plt.plot(xtest,mu, 'b', lw = 3, label = 'opt')
    plt.plot(xtest, mu_diag, 'g', lw=3, label='opt')
    plt.plot(xtest, mu_sup, 'r--', lw=3, label='opt')

    plt.plot(Xtrain,ytrain,'ko', lw = 3)
    plt.plot(xtest,f(xtest),'k--', lw = 3)
    plt.show()