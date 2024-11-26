import cvxpy as cp
import numpy as np
import torch
from scipy.linalg import null_space, orth
from scipy.optimize import minimize
from torch.autograd import grad
from stpy.regression.regularized_dictionary.regularized_dictionary import RegularizedDictionary
from stpy.kernel import KernelFunction
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.regularization.regularizer import L2Regularizer
from stpy.embeddings.embedding_base import EmbeddingBase


class FiniteGaussianLikelihood(RegularizedDictionary):
    '''
        Finite Dictionary Gaussian Process with Gaussian Likelihood
        - more efficient inference with closer form inversion
    '''

    def __init__(self,
                 embedding: EmbeddingBase,
                 s: float = 0.001,
                 lam: float = 1.,
                 d: float = 1,
                 diameter: float = 1.0,
                 bound: float = 1.0,
                 verbose=False,
                 groups=None,
                 bounds=None):

        self.likelihood = GaussianLikelihood(sigma=s)
        self.regularizer = L2Regularizer(lam=lam)
        self.diameter = diameter
        self.s = s
        self.linear_kernel = KernelFunction(kernel_name="linear").linear_kernel
        self.m = embedding.get_m()
        super().__init__(embedding, self.likelihood, regularizer=self.regularizer, d=d, verbose=verbose, bound=bound,
                         groups=groups, bounds=bounds)


    def calculate(self):

        if self.fitted:
            if self.verbose:
                print("Skip fitting.")
                return None
        else:
            if self.verbose:
                pass
                #print ("Fitting.")

            phi = self.embed(self.x)
            if self.weights is None:
                weights = torch.ones(self.x.size()[0]).view(-1,1)
            else:
                weights = self.weights

            #TODO: Change to  Gaussian likelihood call to avoid s and Sigma differences for heteroscedasdic case.
            self.Z = torch.einsum('ij,i,ik->jk', phi, weights.view(-1)/self.s**2, phi) + self.regularizer.hessian(torch.ones(self.m))
            #self.invV = torch.pinverse(self.Z)
            #self.theta_fit = self.invV @ phi.T @ (weights*self.y)/self.s**2
            self.theta_fit = torch.linalg.solve(self.Z, phi.T @ (weights*self.y)/self.s**2)

    def beta(self, delta=0.1, norm=None):
        # self.K = Z_ + self.s * self.s * self.lam * I
        if norm is None:
            norm = self.theta_norm

        if self.beta_fun is None:
            return 2.0

        elif self.beta_fun == "theory":
            K = self.kernel(self.x, self.x) + torch.eye(self.x.size()[0]).double() * self.s ** 2 * self.lam

            beta_value = self.bound * self.lam + torch.logdet(K / ((self.s ** 2) * self.lam)) + 2 * np.log(1 / delta)
            Q = self.embed(self.x)
            Lam = self.lam * torch.eye(self.get_basis_size()).double()
            V = Q.T @ Q / (self.s ** 2) + Lam

            beta_value = self.bound * self.lam + torch.logdet(V) - torch.logdet(Lam) + 2 * np.log(1 / delta)
            beta_value = beta_value
        else:
            return self.beta_fun(self.K, delta=delta, norm=norm)
        return beta_value

    def set_embedding(self, embedding):
        self.embedding = embedding

    def set_m(self, m):
        self.m = m

    def logdet_ratio(self):
        I = torch.eye(int(torch.sum(self.m))).double()
        return torch.logdet(self.K) - torch.logdet(self.s ** 2 * self.lam * I)

    def effective_dim(self, xtest):
        Phi = self.embed(xtest)
        d = torch.trace(torch.solve(Phi.T @ Phi, Phi.T @ Phi + torch.eye(self.get_basis_size()).double() * self.lam)[0])
        return d


    def get_kernel(self):
        embeding = self.embed(self.x)
        Z_ = self.linear_kernel(embeding, embeding)
        K = (Z_ + self.s * self.s * self.lam * torch.eye(int(self.n), dtype=torch.float64))
        return K


    def residuals(self):
        mu, _ = self.mean_std(self.x)
        out = torch.sum((mu - self.y) ** 2)
        return out

    def mean_std(self, xtest):
        '''
            Calculate mean and variance for GP at xtest points
        '''
        embeding = self.embed(xtest)

        # mean
        theta_mean = self.theta_ml()
        ymean = embeding @ theta_mean

        # std
        b = torch.linalg.lstsq(self.invV, embeding)[0]
        diagonal = self.s ** 2 * torch.einsum('ij,ij->i', (embeding, b)).view(-1, 1)
        ystd = torch.sqrt(diagonal)

        return (ymean, ystd)

    def ucb(self, xtest, delta=0.1):
        mu, std = self.mean_std(xtest)
        res = mu + np.sqrt(self.beta(delta=delta)) * std
        return res

    def lcb(self, xtest, delta=0.1):
        mu, std = self.mean_std(xtest)
        res = mu - np.sqrt(self.beta(delta=delta)) * std
        return res

    def sample_matheron(self, xtest, kernel_object, size=1):
        basis = self.get_basis_size()
        zeros = torch.zeros(size=(basis, size), dtype=torch.float64)
        random_vector = torch.normal(mean=zeros, std=1.)

        Z = self.lam * torch.eye(basis, dtype=torch.float64)
        L = torch.linalg.cholesky(Z.transpose(-2, -1).conj()).transpose(-2, -1).conj()
        theta = torch.mm(L, random_vector) + self.prior_mean

        f_prior_xtest = torch.mm(self.embed(xtest), theta)
        f_prior_x = torch.mm(self.embed(self.x), theta)

        K_star = kernel_object.kernel(self.x, xtest)
        N = self.x.size()[0]
        K = kernel_object.kernel(self.x, self.x) + self.s ** 2 * self.lam * torch.eye(N)

        f = f_prior_xtest + K_star @ torch.pinverse(K) @ (self.y - f_prior_x)
        return f

    def sample_theta(self, size=1, prior=False):

        basis = self.get_basis_size()

        zeros = torch.zeros(size=(basis, size), dtype=torch.float64)
        random_vector = torch.normal(mean=zeros, std=1.)
        self.precompute()

        if self.fitted == True and prior == False:
            self.L = torch.linalg.cholesky(self.get_invV()) * self.s
            theta = self.theta_mean()
            theta = theta + torch.mm(self.L, random_vector)
        else:
            Z = self.lam * torch.eye(basis, dtype=torch.float64)
            L = torch.linalg.cholesky(Z.transpose(-2, -1).conj()).transpose(-2, -1).conj()
            theta = torch.mm(L, random_vector) + self.prior_mean

        return theta

    def ucb_optimize(self, beta, multistart=25, lcb=False, minimizer="L-BFGS-B"):

        # precompute important (theta)
        theta_mean, K = self.theta_mean(var=True)

        if lcb == False:
            fun = lambda x: - (self.embed(torch.from_numpy(x).view(1, -1)) @ theta_mean + \
                               beta * torch.sqrt(self.embed(torch.from_numpy(x).view(1, -1)) @ K @ self.embed(
                        torch.from_numpy(x).view(1, -1)).T)).detach().numpy()[0]
        else:
            fun = lambda x: - (self.embed(torch.from_numpy(x).view(1, -1)) @ theta_mean - \
                               beta * torch.sqrt(self.embed(torch.from_numpy(x).view(1, -1)) @ K @ self.embed(
                        torch.from_numpy(x).view(1, -1)).T).detach().numpy()[0]).numpy()[0]

        if self.bounds == None:
            mybounds = tuple([(-self.diameter, self.diameter) for _ in range(self.d)])
        else:
            mybounds = self.bounds

        results = []
        for j in range(multistart):

            x0 = np.random.randn(self.d)
            for i in range(self.d):
                x0[i] = np.random.uniform(mybounds[i][0], mybounds[i][1])

            if minimizer == "L-BFGS-B":
                res = minimize(fun, x0, method="L-BFGS-B", jac=None, tol=0.0001, bounds=mybounds)
                solution = res.x
            else:
                raise AssertionError("Wrong optimizer selected.")

            results.append([solution, -fun(solution)])

        results = np.array(results)
        index = np.argmax(results[:, 1])
        solution = results[index, 0]
        return (torch.from_numpy(solution).view(1, -1), -torch.from_numpy(fun(solution)))

    def sample_and_optimize(self, xtest=None, multistart=25, minimizer="L-BFGS-B", grid=100, verbose=0):
        '''
            Sample functions from Gaussian Process and take Maximum using
            first order maximization
        '''

        # sample linear approximating
        theta = self.sample_theta()

        # get bounds
        if self.bounds == None:
            mybounds = tuple([(-self.diameter, self.diameter) for _ in range(self.d)])
        else:
            mybounds = self.bounds

        fun = lambda x: -torch.mm(torch.t(theta), torch.t(self.embed(torch.from_numpy(x).view(1, -1)))).numpy()[0]

        results = []
        for j in range(multistart):
            x0 = np.random.randn(self.d)
            for i in range(self.d):
                x0[i] = np.random.uniform(mybounds[i][0], mybounds[i][1])

            if minimizer == "L-BFGS-B":
                res = minimize(fun, x0, method="L-BFGS-B", jac=None, tol=0.0001, bounds=mybounds)
                solution = res.x
            else:
                raise AssertionError("Wrong optimizer selected.")

            results.append([solution, -fun(solution)])
        results = np.array(results)
        index = np.argmax(results[:, 1])
        solution = results[index, 0]

        return (torch.from_numpy(solution), -torch.from_numpy(fun(solution)))

    def sample(self, xtest, size=1, prior=False):
        '''
            Sample functions from Gaussian Process
        '''
        theta = self.sample_theta(size=size, prior=prior)
        f = torch.mm(self.embed(xtest), theta)
        return f

    def sample_and_max(self, xtest, size=1):
        '''
            Sample functions from Gaussian Process and take Maximum
        '''
        f = self.sample(xtest, size=size)
        index = np.argmax(f, axis=0)
        return (xtest[index, :], f[index, :])


if __name__ == "__main__":
    N = 10
    s = 0.1
    n = 256
    L_infinity_ball = 0.5

    d = 1
    m = 128

    xtest = torch.from_numpy(interval(n, d, L_infinity_ball=L_infinity_ball))
    x = torch.from_numpy(np.random.uniform(-L_infinity_ball, L_infinity_ball, N)).view(-1, 1)

    F_true = lambda x: torch.sin(x * 4) ** 2 - 0.1
    F = lambda x: F_true(x) + s * torch.randn(x.size()[0]).view(-1, 1).double()
    y = F(x)

    emb = RFFEmbedding(m=m, gamma=0.1)
    Reggr = KernelizedFeatures(embedding=emb, m=m, d=1)
    Reggr.fit_gp(x, y)
    Reggr.visualize(xtest, f_true=F_true)
