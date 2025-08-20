from stpy.regression.finite_gaussian_process import FiniteGaussianLikelihood
from stpy.regression.regularized_dictionary.regularized_dictionary import RegularizedDictionary
from stpy.regularization.regularizer import L2Regularizer
from stpy.regularization.sdp_constraint import SDPConstraint
from stpy.kernel import KernelFunction
import torch
from torchmin import minimize as minimize_torch
from stpy.candidate_set import CandidateDiscreteSet
import numpy as np
from autograd_minimize import minimize


class ConvexRKHS(FiniteGaussianLikelihood):

    def __init__(self, *args,
                 regularizer_scale=SDPConstraint(trace_constraint=0.0001),
                 Gamma=None,
                 typ="ard",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.m = self.embedding.get_m()
        self.anchor = None

        if Gamma is None:
            self.Gamma = None
        else:
            self.Gamma = Gamma

        self.ard_type = typ
        if typ == "cov":
            if self.Gamma is None:
                self.Gamma = torch.eye(self.m, self.m).double()
            self.kernel = KernelFunction(d=self.m, kernel_name='full_covariance_se', cov=self.Gamma)
        
        elif typ == "ard":
            if self.Gamma is None:
                self.Gamma = torch.ones(size = (self.m,1)).double().view(-1)
            self.kernel = KernelFunction(d=self.m, kernel_name='ard', ard_gamma=self.Gamma)
        
        elif typ == "se":
            if self.Gamma is None:
                self.Gamma = torch.eye(1,1).double().view(-1)
            self.kernel = KernelFunction(d=self.m, kernel_name='squared_exponential', gamma=self.Gamma)
        
        elif typ == "linear":
            if self.Gamma is None:
                self.Gamma = torch.eye(self.m, self.m).double()
            self.kernel = KernelFunction(d=self.m, kernel_name='linear', cov=self.Gamma)

        elif typ == "linear-norm":
            if self.Gamma is None:
                self.Gamma = torch.eye(self.m, self.m).double()
            self.kernel = KernelFunction(d=self.m, kernel_name='linear_norm', kappa = self.Gamma)
        else:
            raise NotImplementedError("..")
        self.regularizer_scale = regularizer_scale
        self.kernel_object = KernelFunction(d=self.m, kernel_name='linear')

    def get_lam(self):
        return self.regularizer.lam


    def set_kernel(self,Gamma):
        self.Gamma = Gamma
        if  self.ard_type == "cov":
            self.kernel.cov = Gamma
        elif self.ard_type == "ard":
            self.kernel.ard_gamma = Gamma
        elif self.ard_type == "se":
            self.kernel.gamma = Gamma
        elif self.ard_type == "linear":
            self.kernel.cov = Gamma
        elif self.ard_type == "linear-norm":
            self.kernel.kappa = Gamma
        else:
            raise NotImplementedError("..")


    def local_fit(self, weights):
        self.load_data((self.x, self.y), weights=weights)
        self.fitted = False
        super().fit()
        return self.theta_fit

    def optimize_params(self, type='bandwidth',
                        restarts=10,
                        maxiter=1000,
                        mingradnorm=1e-4,
                        verbose=False,
                        scale=10.,
                        bounds=None,
                        parallel=False,
                        cores=None,
                        fit_type='ignore-base',
                        optimizer = 'torchmin'):
        x_data = self.x
        y_data = self.y
        Phi = lambda x: self.embedding.embed(x)
        m = self.m

        def total_loss(Gamma):
            weights = []
            predictions = []

            self.set_kernel(Gamma)
            if self.ard_type == 'se':
                reg = self.regularizer_scale.eval(Gamma.view(1,1) ** 2)
            elif self.ard_type == 'ard':
                reg = self.regularizer_scale.eval(torch.diag(Gamma) ** 2)
            else:
                reg = self.regularizer_scale.eval(Gamma ** 2)

            mu = self.mean(self.x, fit_type=fit_type)

            loss = reg + torch.sum((mu - self.y) ** 2)
            return loss

        # optimize this
        vals = []
        args = []
        for _ in range(restarts):

            if self.ard_type == 'se':
                Gamma = torch.randn((1, 1), requires_grad=True).double().view(-1) * scale
            elif self.ard_type == 'ard':
                Gamma = torch.randn((m, 1), requires_grad=True).double().view(-1) * scale
            elif self.ard_type == 'cov':
                Gamma = torch.randn((m, m), requires_grad=True).double() * scale
            elif self.ard_type == 'linear-norm':
                Gamma = torch.randn((1, 1), requires_grad=True).double() * scale
            else:
                raise NotImplementedError(".")
            #try:
            if optimizer == 'torchmin':
                result = minimize_torch(total_loss, Gamma, method='l-bfgs', disp=verbose + 1)
                vals.append(result.fun)
                args.append(result.x)

            elif optimizer == 'autograd':
                result = minimize(total_loss, Gamma.detach().numpy(), backend='torch', method='L-BFGS-B',
                               precision='float64', tol=1e-6,
                               options={'ftol': 1e-6,
                                        'gtol': mingradnorm, 'eps': 1e-06,
                                        'maxfun': 15000, 'maxiter': maxiter,
                                        'maxls': 20, 'disp': verbose + 1})

                vals.append(float(result.fun))
                args.append(torch.from_numpy(result.x))

            else:
                raise NotImplementedError("Optimizer not implemented.")

            # vals.append(result.fun)
            # args.append(result.x)

        # except Exception as e:
        #     print("Optimization failed.", e)
        Gamma = args[np.argmin(vals)]

        self.set_kernel(Gamma)
        self.Gamma = Gamma
        print("Found optimal:", self.Gamma)

    def get_weights(self, xtest, x=None):
        phitest = self.embed(xtest)

        if x is None:
            x = self.x
            phi = self.phi
        else:
            phi = self.embed(x)

        out = torch.zeros(size=(phitest.size()[0], x.size()[0])).double()
        for i, x in enumerate(phitest):
            # construct weighting
            self.set_kernel(self.Gamma)
            out[i, :] = self.kernel.get_kernel_internal()(x.view(1, -1), phi).view(-1)
        return out

    def std(self, xtest):
        phitest = self.embed(xtest)
        out = torch.zeros(size=(phitest.size()[0], 1)).double()
        for i, x in enumerate(phitest):
            # construct weighting
            self.set_kernel(self.Gamma)

            w = self.kernel.get_kernel_internal()(x.view(1, -1), self.phi)

            # create a local fit
            self.local_fit(w)

            # local model
            std = np.sqrt(x.T @ torch.linalg.solve(self.Z, x))
            # save
            out[i] = std
        return out

    def mean_std(self, xtest):
        phitest = self.embed(xtest)
        out = torch.zeros(size=(phitest.size()[0], 1)).double()
        stds = torch.zeros(size=(phitest.size()[0], 1)).double()
        for i, x in enumerate(phitest):
            # construct weighting
            self.set_kernel(self.Gamma)

            w = self.kernel.get_kernel_internal()(x.view(1, -1), self.phi)

            # create a local fit
            self.local_fit(w)

            # local model
            f = x @ self.theta_fit
            std = np.sqrt(x.T @ torch.linalg.solve(self.Z, x))
            # save
            stds[i] = std
            out[i] = f

        return out,stds

    def model_similarity(self, xtest):
        if self.anchor is None:
            raise ValueError("Anchor is not set; local model is not specified.")
        else:
            phitest = self.embed(xtest)
            w = self.kernel.get_kernel_internal()(self.anchor.view(1, -1), phitest)
            D = torch.sqrt(torch.diag(w.view(-1)))
            K = D@phitest@phitest.T@D
            return K

    def mean_iterative(self, xtest, fit_type="cutoff", tol=10e-5, cutoff = 0.01):
        phitest = self.embed(xtest)
        out = torch.zeros(size=(phitest.size()[0], 1)).double()
        self.set_kernel(self.Gamma)
        
        for i, x in enumerate(phitest):

            # construct weighting 
            w = self.kernel.get_kernel_internal()(x.view(1, -1), self.phi)

            if fit_type == 'ignore-base':
                # remove the true point from the fitting
                w[w > 1 - tol] = 0.

            elif fit_type == 'cutoff':
                w[w <cutoff] = 0.
                w[w >= cutoff] = 1.
            else:
                pass 

            # create a local fit
            self.local_fit(w)

            # local model
            f = x @ self.theta_fit
            # save
            out[i] = f
        return out
    
    def mean(self, xtest, fit_type = 'ignore-base', tol = 10e-5, cutoff = 0.01):
        phitest = self.embed(xtest)
        out = torch.zeros(size=(phitest.size()[0], 1)).double()
        self.set_kernel(self.Gamma)
        W = self.kernel.get_kernel_internal()(phitest, self.phi)
        if fit_type == 'ignore-base':
            # remove the true point from the fitting
            W[W > 1 - tol] = 0.

        elif fit_type == 'cutoff':
            W[W < cutoff] = 0.
            W[W >= cutoff] = 1.
        else:
            pass 

        d = phitest.size()[1]
        covar_matrix = torch.einsum('ij,jk,jl->kil' ,(self.phi.T,W,self.phi)) + self.regularizer.lam * torch.eye(d).double().unsqueeze(0)
        b = torch.einsum('ik,kl,kp->lip', (self.phi.T, W, self.y))#.squeeze(-1)
        theta = torch.linalg.solve(covar_matrix, b)
        out = torch.einsum('ij,ijp->ip', (phitest, theta))
        return out 

    def best_points_so_far(self):
        """
        get all points which are above max - 2*s
        :return:
        """
        conservative_best_value = torch.max(self.y) - 2 * self.s
        mask = self.y > conservative_best_value
        return self.x[mask, :]

    def sample_neighbourhood_sample(self, x_loc, candidate_set, cut_off=0.01, size=10):
        if isinstance(CandidateDiscreteSet, candidate_set):
            xtest = self.embed(candidate_set.get_options_raw)
            w = self.weight_scaling(self.Gamma, 1., x_loc, xtest, self.embed)
            selection = xtest[w > cut_off]
            max_v = selection.size()[0]
            indices = np.random.choice(max_v, size=size)
            out = selection[indices]
            return out
        elif isinstance(ConditionalGenerativeModel, candidate_set):
            pass
        else:
            NotImplementedError("The requested candidate set method is not implemented")

    def func_gradient(self, x):
        w = self.weight_scaling(self.Gamma, 1., x, self.x, self.embed)
        return self.local_fit(weights=w)


if __name__ == "__main__":
    from stpy.embeddings.polynomial_embedding import ChebyschevEmbedding
    from stpy.probability.gaussian_likelihood import GaussianLikelihood
    from stpy.regularization.regularizer import L2Regularizer
    from stpy.helpers.helper import interval_torch
    import matplotlib.pyplot as plt
    import numpy as np
    import time 

    embedding = ChebyschevEmbedding(p=4, d=1)
    n = 256
    N = 4
    lam = 1
    s = 0.1
    Reg = SDPConstraint(trace_constraint=0.0000001)
    Estimator = ConvexRKHS(embedding,
                           typ="ard",
                           s=s,
                           lam=lam,
                           verbose=True,
                           Gamma=None,
                           regularizer_scale=Reg
                           )

    xtest = interval_torch(d=1, n=n)
    x = torch.zeros(size=(N, 1)).double()
    x = x.uniform_()

    gamma_original = torch.randn(size=(embedding.get_m(),)).double()
    Phi_original = lambda x: embedding.embed(x) @ torch.diag(gamma_original)
    Phi = lambda x: embedding.embed(x)
    y = torch.sum(Phi_original(x) ** 2, axis=1).view(-1, 1)
    ytest = torch.sum(Phi_original(xtest) ** 2, axis=1).view(-1, 1)

    Estimator.load_data((x, y))
    

    t1 = time.time()
    out = Estimator.mean(xtest, fit_type='nothing')
    t2 = time.time()
    out2 = Estimator.mean_iterative(xtest, fit_type='nothing')
    t3 = time.time()
    print("Time for local fit:", t2 - t1)
    print("Time for linear fit:", t3 - t2)
    print(torch.sum((out-out2)**2))
    mu,std= Estimator.mean_std(xtest)

    Estimator.optimize_params(verbose=True,optimizer="torchmin",
                              restarts=5)

    mu2 = Estimator.mean(xtest, fit_type = 'nothing')
    std2 = Estimator.std(xtest)
    print("True gamma:", gamma_original)
    print("Optimized gamma:", torch.diag(Estimator.Gamma))
    print("Optimized gamma:", torch.diag(Estimator.Gamma))

    # offset = 20
    # Phi = lambda x: embedding.embed(x)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.plot(xtest, mu.detach(), 'b', label='original hyperparams')
    ax1.plot(xtest, out.detach(), 'y--', label='test')

    ax1.fill_between(xtest.view(-1), mu.view(-1) - std.view(-1), mu.view(-1) + std.view(-1),color =  'b', alpha = 0.1)

    ax1.plot(xtest, mu2.detach(), 'g', label='optimized hyperparams')
    ax1.fill_between(xtest.view(-1), (mu2 - std2).view(-1), (mu2 + std2).view(-1), color = 'g', alpha = 0.1)

    ax1.plot(xtest, ytest, 'k--', label='true function')
    ax1.plot(Estimator.x, Estimator.y, 'ko', label='data points')

    # Pick a random point in the interval
    x_test = torch.tensor([[0.5]])
    w = Estimator.get_weights(x_test, x=xtest)

    ax1.plot(xtest.view(-1), w.view(-1), 'r--')
    ax1.plot(x_test.view(-1), [1.0], 'ro')

    #
    ax1.legend()
    plt.show()
