from stpy.regression.finite_gaussian_process import FiniteGaussianLikelihood
from stpy.regression.regularized_dictionary.regularized_dictionary import RegularizedDictionary
from stpy.regularization.regularizer import L2Regularizer
from stpy.regularization.sdp_constraint import SDPConstraint
from stpy.kernel import KernelFunction
import torch
from torchmin import minimize
from stpy.candidate_set import CandidateDiscreteSet


class ConvexRKHS(FiniteGaussianLikelihood):

    def __init__(self, *args,
                 regularizer_scale=SDPConstraint(trace_constraint = 0.0001),
                 Gamma = None,
                 ARD = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.m = self.embedding.get_m()
        if Gamma is None:
            self.Gamma = torch.eye(self.m, requires_grad=True).double()
        else:
            self.Gamma = Gamma
        if ARD:
            self.kernel = KernelFunction(d=self.m, kernel_name='full_covariance_se', cov=self.Gamma)
        else:
            self.kernel = KernelFunction(d=self.m, kernel_name='linear', cov=self.Gamma)
        self.regularizer_scale = regularizer_scale

    def local_fit(self, weights):
        self.load_data((self.x, self.y), weights=weights)
        #print ("weights:",self.weights.T)
        self.fitted = False
        super().fit()
        #print ("fit:",self.theta_fit.T)
        return self.theta_fit

    def optimize_params(self, type='bandwidth',
                        restarts=10,
                        maxiter=1000,
                        mingradnorm=1e-4,
                        verbose=False,
                        scale=1.,
                        bounds=None,
                        parallel=False,
                        cores=None,
                        diagonal=True):
        x_data = self.x
        y_data = self.y
        Phi = lambda x: self.embedding.embed(x)
        m = self.m

        def total_loss(Gamma):
            weights = []
            predictions = []

            if diagonal:
                diagonal_mask = torch.eye(Gamma.size(0), Gamma.size(1),
                                          dtype=torch.float32,
                                          device=Gamma.device)
                self.Gamma = torch.diag(torch.diag(Gamma)**2)
                #invGamma = torch.diag(1. / torch.diag(self.Gamma))
            else:
                self.Gamma = Gamma
                #invGamma = torch.diag(1. / torch.diag(self.Gamma.T@self.Gamma))

            mu = self.mean(self.x)
            loss = self.regularizer_scale.eval( self.Gamma) + torch.sum((mu - self.y) ** 2)
            return loss

        # optimize this
        vals = []
        args = []
        for _ in range(restarts):
            Gamma = torch.randn((m, m), requires_grad=True).double()
            result = minimize(total_loss, Gamma, method='bfgs', disp=2)
            vals.append(result.fun)
            args.append(result.x)

        self.Gamma = args[np.argmin(vals)]
        print ("Found optimal:", self.Gamma)

    def get_weights(self, xtest, x = None):
        phitest = self.embed(xtest)
        if x is None:
            x = self.x
            phi = self.phi
        else:
            phi = self.embed(x)
        out = torch.zeros(size=(phitest.size()[0], x.size()[0])).double()
        print (out.size())
        for i, x in enumerate(phitest):
            # construct weighting
            self.kernel.cov = self.Gamma
            out[i,:] = self.kernel.get_kernel_internal()(x.view(1, -1), phi).view(-1)
        return out

    def mean(self, xtest):
        phitest = self.embed(xtest)
        out = torch.zeros(size=(phitest.size()[0], 1)).double()
        for i, x in enumerate(phitest):
            # construct weighting
            self.kernel.cov = self.Gamma
            w = self.kernel.get_kernel_internal()(x.view(1, -1), self.phi)

            # create a local fit
            self.local_fit(w)

            # local model
            f = x @ self.theta_fit

            # save
            out[i] = f
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

    embedding = ChebyschevEmbedding(p=4, d=1)
    n = 256
    N = 4
    lam = 1
    s = 0.1
    Estimator = ConvexRKHS(embedding, ARD = True, s=s, lam=lam, verbose=True, Gamma = 0.5*torch.eye(embedding.get_m()).double())

    xtest = interval_torch(d=1, n=n)
    x = torch.zeros(size=(N, 1)).double()
    x = x.uniform_()

    gamma_original = torch.randn(size=(embedding.get_m(),)).double()
    Phi_original = lambda x: embedding.embed(x) @ torch.diag(gamma_original)
    Phi = lambda x: embedding.embed(x)
    y = torch.sum(Phi_original(x) ** 2, axis=1).view(-1, 1)
    ytest = torch.sum(Phi_original(xtest) ** 2, axis=1).view(-1, 1)

    Estimator.load_data((x, y))
    mu = Estimator.mean(xtest).clone()

    Estimator.optimize_params()
    mu2 = Estimator.mean(xtest)

    # Estimator.optimize_params()
    #

    print ("True gamma:",gamma_original)
    print ("Optimized gamma:", torch.diag(Estimator.Gamma))

    # offset = 20
    # Phi = lambda x: embedding.embed(x)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.plot(xtest, mu.detach(), 'b', label = 'original hyperparams')
    ax1.plot(xtest, mu2.detach(), 'g', label = 'optimized hyperparams')
    ax1.plot(xtest, ytest, 'k--', label = 'true function')
    ax1.plot(Estimator.x, Estimator.y, 'ko', label = 'data points')

    # Pick a random point in the interval
    x_test = torch.tensor([[0.5]])
    w = Estimator.get_weights(x_test, x = xtest)

    ax1.plot(xtest.view(-1), w.view(-1),'r--')
    ax1.plot(x_test.view(-1), [1.0],'ro')

    #
    ax1.legend()
    plt.show()
