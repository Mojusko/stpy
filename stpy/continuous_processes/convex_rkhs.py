from stpy.continuous_processes.kernelized_features import KernelizedFeatures
import torch
from torchmin import minimize
from stpy.candidate_set import CandidateDiscreteSet
from stpy.generative_models.conditional_generative_model import ConditionalGenerativeModel
class ConvexRKHS(KernelizedFeatures):
    """
    """

    def __init__(self, embedding, m, lam = 0. , s = 0.01):
        super().__init__(embedding, m)
        self.Gamma = torch.eye(m, requires_grad=True).double()
        self.lam = lam
        self.s = s
    def fit(self,x=None,y=None):
        """
        legacy method
        :param x:
        :param y:
        :return:
        """
        pass
    def weight_scaling(self, Gamma, scale, x_single, y, Phi):
        x = torch.tile(x_single, (y.size()[0], 1))
        return torch.exp(-torch.sum(((Phi(x) - Phi(y)) @ Gamma /scale) ** 2, axis=1))

    def local_fit(self, weights):
        D = torch.diag(weights)
        X = self.embed(self.x)
        theta = torch.linalg.inv((X.T @ D @ X) + self.lam * torch.eye(self.m)) @ X.T @ D @ self.y
        return theta
    def optimize_params(self, type='bandwidth', restarts=10, regularizer=None,
						maxiter=1000, mingradnorm=1e-4, verbose=False, optimizer="pymanopt", scale=1., weight=1., save = False,
								save_name = 'model.np', init_func = None, bounds = None, parallel = False, cores = None):

        x_data = self.x
        y_data = self.y
        Phi = lambda x: self.embedding.embed(x)
        m = self.get_basis_size()


        def total_loss(gamma):
            weights = []
            predictions = []
            for i in range(x_data.size()[0]):
                x = x_data[i]
                Gamma =  torch.diag(gamma)
                w = self.weight_scaling(Gamma, 1., x, x_data, Phi)
                X = Phi(x_data)

                # local fit in the new coordinates
                theta = self.local_fit(w)

                # prediction
                predictions.append(X @ theta)

                # weights determining the importance of the predictions
                weights.append(w)

            loss = 0

            for p1, w1 in zip(predictions, weights):
                # loss that makes sure we predict correctly
                loss = 1* torch.sum(((p1 - y_data) ** 2)/(self.s**2) * (w1)) / 2

                for p2, w2 in zip(predictions, weights):
                    # loss that makes sure the predictions are consistent (this can be a larger set)
                    loss += 1* torch.sum((p1 - p2)**2/(self.s**2) * (w1 * w2))

            return loss + 0.001*torch.sum(gamma**2)

        # optimize this
        vals = []
        args = []
        for _ in range(restarts):
            gamma = torch.randn(m, requires_grad=True).double()**2
            total_loss(gamma)
            result = minimize(total_loss, gamma, method='bfgs', disp=2)
            vals.append(result.fun)
            args.append(result.x)

        self.Gamma = torch.diag(args[np.argmin(vals)])

    def mean(self, xtest):
        phitest = self.embed(xtest)
        out = torch.zeros(size = (phitest.size()[0],1)).double()
        for i, x in enumerate(xtest):
            w = self.weight_scaling(self.Gamma, 1., x, self.x, self.embed)
            out[i] = 0.
            f = self.embed(x)@self.local_fit(w)
            out[i] = f
        return out

    def best_points_so_far(self):
        """
        get all points which are above max - 2*s
        :return:
        """
        conservative_best_value = torch.max(self.y) - 2*self.s
        mask = self.y > conservative_best_value
        return self.x[mask,:]

    def sample_neighbourhood_sample(self, x_loc, candidate_set, cut_off = 0.01, size = 10):
        if isinstance(CandidateDiscreteSet,candidate_set):
            xtest = self.embed(candidate_set.get_options_raw)
            w = self.weight_scaling(self.Gamma, 1., x_loc,xtest, self.embed)
            selection = xtest[w > cut_off]
            max_v = selection.size()[0]
            indices = np.random.choice(max_v, size = size)
            out = selection[indices]
            return out
        elif isinstance(ConditionalGenerativeModel, candidate_set):
            pass
        else:
            NotImplementedError("The requested candidate set method is not implemented")

    def func_gradient(self, x):
        w = self.weight_scaling(self.Gamma, 1., x, self.x,  self.embed)
        return self.local_fit(weights=w)


if __name__ == "__main__":
    from stpy.embeddings.polynomial_embedding import ChebyschevEmbedding
    from stpy.helpers.helper import interval_torch
    import matplotlib.pyplot as plt
    import numpy as np

    embedding = ChebyschevEmbedding(p=4, d=1)
    n = 256
    N = 4
    lam = 1e-6
    gamma_original = torch.randn(size = (embedding.get_m(),)).double()
    xtest = interval_torch(d=1, n=n)
    x = torch.zeros(size =(N,1)).double()
    x = x.uniform_()

    Phi_original = lambda x: embedding.embed(x) @ torch.diag(gamma_original)
    Phi = lambda x: embedding.embed(x)
    y = torch.sum(Phi_original(x) ** 2, axis=1).view(-1)
    ytest= torch.sum(Phi_original(xtest) ** 2, axis=1).view(-1)
    Estimator = ConvexRKHS(embedding, embedding.get_m(), lam = lam )
    #Estimator = torch.compile(Estimator)

    Estimator.load_data((x, y))
    Estimator.optimize_params()

    print ("True gamma:",gamma_original)
    print ("Optimized gamma:", torch.diag(Estimator.Gamma))
    offset = 20
    Phi = lambda x: embedding.embed(x)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    for i in range(xtest.size()[0]):
        x = xtest[i]
        w = Estimator.weight_scaling(Estimator.Gamma,  1., x, xtest, Phi)
        D = torch.diag(w)
        X = Phi(xtest)
        theta = torch.linalg.inv((X.T@D@X) + lam * torch.eye(embedding.get_m()))@X.T@D@ytest
        prediction = (X@theta).detach()

        if i%64 == 0:
            p = ax1.plot(xtest[i],
                     prediction[i],'o',ms = 10)

            ax1.plot(xtest[np.max([0,i-offset]):np.min([i+offset,n])],
                     prediction[np.max([0,i-offset]):np.min([i+offset,n])], color = p[0].get_color())
            ax2.plot(xtest, w, color = p[0].get_color())

    mu = Estimator.mean(xtest)

    ax1.plot(xtest, mu, 'b')
    ax1.plot(xtest,ytest,'k--')
    ax1.plot(Estimator.x,Estimator.y,'ko')

    plt.show()