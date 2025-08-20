import cvxpy as cp
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from stpy.estimator import Estimator
import torch.nn.init as init
import torch.func as func


class MLPRegressor(nn.Module, Estimator):

    def __init__(self, layer_sizes,
                 likelihood,
                 regularizer,
                 activation=nn.ReLU,
                 output_activation=None,
                 learning_rate=0.0001,
                 verbose=True,
                 epochs=100,
                 batch_size=32,
                 device = 'cpu'
                 ):
        """
        Multi-Layer Perceptron (MLP) for regression with likelihood and regularization.
        :param layer_sizes: List containing the sizes of each layer, including input and output sizes.
        :param likelihood: A likelihood object implementing probabilistic evaluation.
        :param regularizer: A regularization object for model constraints.
        :param activation: Activation function for hidden layers (default: ReLU).
        :param output_activation: Activation function for the output layer (default: None for regression).
        :param learning_rate: Learning rate for the optimizer.
        """
        super(MLPRegressor, self).__init__()

        self.likelihood = likelihood
        self.regularizer = regularizer
        self.verbose = verbose
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size

        layers = []
        for i in range(len(layer_sizes) - 1):
            # iteratively adds
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            # always acts on the last one
            init.normal_(layers[-1].weight)
            layers[-1].weight.values =layers[-1].weight/ (10*torch.sum((layers[-1].weight)**2)*len(layer_sizes))
            init.zeros_(layers[-1].bias)

            if i < len(layer_sizes) - 2:
               # layers.append(nn.LayerNorm(layer_sizes[i + 1]))  # Normalizing activations
                layers.append(activation())

        if output_activation is not None:
            layers.append(output_activation())

        self.model = nn.Sequential(*layers).to(self.device)

        self.optimizer = torch.optim.LBFGS(self.parameters(), lr=learning_rate)
        #self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        # use for interior point solver for model constraints
        self.lam2 = 0.1
        self.bound = 1.
        self.lam1 = 10.
    def forward(self, x):
        return self.model(x)

    def config(self, batch_size=32, epochs=100, verbose=True):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

    def load_data(self, d):
        self.x = d[0]
        self.y = d[1]
        self.fitted = False

    def fit(self):

        if self.fitted == True:
            return None

        dataset = TensorDataset(self.x, self.y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.likelihood.load_data((self.x.to(self.device), self.y.to(self.device)))
        self.x = self.x.to(self.device)
        for epoch in range(self.epochs):

            def closure():
                self.optimizer.zero_grad()  # Clear previous gradients
                total_loss = self.likelihood.evaluate_log(self.forward(self.x))

                # this is the constraint on the norm
                total_loss += -self.lam2 * torch.log(-(self.regularizer.eval(list(self.parameters())) - self.bound))

                total_loss.backward()
                return total_loss

            self.optimizer.step(closure)  # LBFGS requires a closure
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {closure().item() / len(dataloader)}")

        self.fittted = True
        self.theta_ml = copy.deepcopy(self.model.state_dict())

    def map(self, x):
        with torch.no_grad():
            return self.forward(x.to(self.device))

    def evaluate(self, x_test, y_test):
        with torch.no_grad():
            predictions = self.forward(x_test)
            loss = self.likelihood.evaluate_log_custom(predictions, y_test)
        return loss

    def optimize_functional_cvxpy_linear(self, xtest, delta = 0.1, loss_landscape = False, type ='ucb'):
        n = len(xtest)

        if type == 'lcb':
            sign = -1
        elif type == 'ucb':
            sign = 1

        # comparator likelihood
        base = self.likelihood.evaluate_log(self.model(self.x.to(self.device)))

        def pred(params, x):
            return func.functional_call(self.model, (params, buffers), x.to(self.device))

        def loss_points(params,x):
            return self.likelihood.evaluate_log_vector(pred(params, x))

        # getting the optimized parameter dictionary
        params = {name: param.clone().detach() for name, param in self.theta_ml.items()}
        buffers = {name: b.detach().clone() for name, b in self.model.named_buffers()}

        # calculate gradient w.r.t. to theta at specific x, and embed
        phi = lambda x: torch.hstack([j.flatten(start_dim = 1) for j in  func.jacfwd(pred, argnums=0)(params, x).values()]) * self.learning_rate
        b = lambda x: pred(params, x)

        # number of parameters
        m = sum(p.numel() for p in self.model.parameters())

        # embedding of the datapoints
        X = phi(self.x)

        # gradients of individual log-losses of each point
        if loss_landscape:
            Phi = torch.hstack([j.flatten(start_dim = 1) for j in  func.jacfwd(loss_points, argnums=0)(params, x).values()])
            N = Phi.size()[0]

        ucbs = []
        for i in range(n):

            xx = xtest[i].view(1, -1)
            Phi_i = phi(xx).view(1, -1)

            if not loss_landscape:
                z = cp.Variable((m, 1))

                y_pred = b(xx).detach() + Phi_i.detach() @ z #+ cp.sum_squares(z)*0.5*self.learning_rate
                likelihood_func = self.likelihood.get_objective_cvxpy_linear(f=lambda x: b(self.x).detach() + X.detach() @ z)(None)
                constraints = []

            else:
                z = cp.Variable((N, 1))

                y_pred = b(xx).detach() + Phi_i.detach() @ Phi.detach().T @ z  # + cp.sum_squares(z)*0.5*self.learning_rate

                likelihood_func = self.likelihood.get_objective_cvxpy_linear(
                f=lambda x: b(self.x).detach() + X.detach() @ Phi.detach().T @ z)(None)

                # one gradient update?
                constraints = [cp.sum_squares(z) <= N]

            # This would be ideal with smoothness property but can be controlled anyway
            #constraints = [u >= cp.sum(cp.square(z))*0.5*self.learning_rate]

            constraints += [likelihood_func <= base.detach() + np.log(1.0/delta)]

            # this is artificial bound; and should rather depend on the total norm of the parameters
            #constraints += [cp.sum_squares(z) <= self.bound]

            objective = cp.Maximize(sign*y_pred)
            prob = cp.Problem(objective, constraints)
            prob.solve()
            ucbs.append(sign*prob.value)

        return torch.Tensor(ucbs).double()





    def optimize_functional_linear(self, x, delta=0.1, type = 'max'):
        lam = self.lam1
        lam2 = self.lam2


        #### new model is
        ## f_base(x) - Phi(x)^z + u
        ## u <= L h^2 ||z||^2
        ## h approx 1/L; L is the Lipschitz constant; or 1 over stepsize
        ## u < (1/L)*||z||^2;

        ## Newton method update is also reasonable

        base = self.likelihood.evaluate_log(self.model(self.x.to(self.device)))

        def loss_fn(params, buffers, constraints = False):

            model_out = func.functional_call(self.model, (params, buffers), xtest.to(self.device))
            test_likelihood = self.likelihood.evaluate_log(func.functional_call(self.model, (params, buffers), self.x))

            if type == 'max':
                out = -torch.logsumexp(model_out, 0)
            elif type == 'ucb':
                out = -model_out
            elif type == 'lcb':
                out = model_out
            else:
                raise NotImplementedError("This functional is not known:", type)

            constraint1 = (-test_likelihood + base + torch.log(torch.tensor(1.0 / delta)))
            constraint2 = self.regularizer.eval(list(params.values())) - self.bound

            if not constraints:
                return out - lam * constraint1 - lam2 * torch.log(-constraint2)
            else:
                return out ,-constraint1 ,-constraint2

        params = {name: torch.nn.Parameter(p.clone()) for name, p in self.model.named_parameters()}
        buffers = {name: b.detach().clone() for name, b in self.model.named_buffers()}

#        optimizer = optim.Adam(params.values(), lr=self.learning_rate)
        optimizer = torch.optim.LBFGS(params.values(), lr=self.learning_rate)

        def closure():
            optimizer.zero_grad()
            total_loss = loss_fn(params, buffers)
            total_loss.backward()
            return total_loss

        for epoch in range(self.epochs):
            optimizer.step(closure)  # LBFGS step with closure

            out, constraint1, constraint2 = loss_fn(params, buffers, constraints = True)

            if self.verbose and (epoch + 1) % 10 == 0:
                print (f"Calculating: {type}, with loss: {closure().item()}, "
                       f"constraints: {constraint1}, {constraint2},"
                       f" loss: {out}")

        if type == 'max':
            return -loss_fn(params, buffers),  func.functional_call(self.model, (params, buffers), xtest)

        else:
            return loss_fn(params, buffers)

    def ucb(self, x, delta=0.1, loss_landscape = False):
        return self.optimize_functional_cvxpy_linear(x, delta, type='ucb', loss_landscape=loss_landscape)

    def lcb(self, x, delta=0.1, loss_landscape=False):
        return self.optimize_functional_cvxpy_linear(x, delta, type='lcb', loss_landscape=loss_landscape)

    def maximum(self, x, delta=0.1):
        return self.optimize_functional(x, delta, type='max')


if __name__ == "__main__":
    from stpy.probability.gaussian_likelihood import GaussianLikelihood
    from stpy.regularization.regularizer import L2Regularizer
    import matplotlib.pyplot as plt

    # Example usage

    N = 100
    sigma = 0.1
    x = torch.randn(N, 1)
    eps = torch.randn(N, 1) * sigma
    y = torch.sin(x) + eps
    hidden = 32
    mlp = MLPRegressor([1, hidden, hidden, 1],
                       likelihood=GaussianLikelihood(sigma=sigma),
                       regularizer=L2Regularizer(lam = 0.01),
                       epochs=150,
                       learning_rate=1e-5)

    mlp.load_data((x, y))
    mlp.fit()

    xtest = torch.linspace(-3, 3, 50).view(-1, 1)

    plt.plot(xtest, torch.sin(xtest), 'g-')
    plt.plot(xtest, mlp.map(xtest).to('cpu').detach().numpy(), 'b-')

    plt.plot(xtest, mlp.ucb(xtest).to('cpu').detach().numpy(), 'r--')
    plt.plot(xtest, mlp.ucb(xtest, loss_landscape=True).to('cpu').detach().numpy(), 'orange')

    plt.plot(x, y, 'ko')

    plt.show()

