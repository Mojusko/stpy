import argparse
import numpy as np
import torch

import matplotlib.pyplot as plt
from stpy.regression.regularized_dictionary import RegularizedDictionary
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.regularization.regularizer import L2DiagRegularizer
from stpy.embeddings.polynomial_embedding import CustomEmbedding
from stpy.probability.noise_models import LaplaceNoise, GaussianNoise, HuberNoise
from stpy.helpers.helper import interval_torch, cartesian
from stpy.kernels import KernelFunction
sigma = 0.1
d = 2
Lam = torch.eye(d)
bound = 1.

theta = torch.Tensor([[1.0, 1.0]]).double().T
f = lambda x: x @ theta

#x = torch.randn(10, 2).double()
x = torch.Tensor([[1.0, 1.0],[1.0, 1.0],[1.0, 0.0],[-1.0, -1.0]]).double()
# x = interval_torch(2, 2)
print(x)
y = f(x) + sigma * torch.randn(x.size()[0], 1).double()

bound1 = []
bound2 = []
bound3 = []
test = torch.Tensor([[0.0, 1.0],[1.0, 0.0]]).double()

weightings = ["none", "bias", "custom"]
for checktype in weightings:
    likelihood = GaussianLikelihood(sigma=sigma)

    regularizer = L2DiagRegularizer(Lam=Lam)

    embedding_function = lambda x: x
    embedding = CustomEmbedding(2, embedding_function,2)


    e1 = torch.Tensor([[1.0, 0.0]]).double()
    print (e1.size())
    def check_fn(obj,x):
        try:
            V = obj.theta_covar()
            num = (e1 @ torch.inverse(V) @ x.T)**2
            denom = (e1 @ torch.inverse(V) @ e1.T)**2
            val = denom/num
            if val < 0.999:
                return 0.
            else:
                return 1.
        except:
            return 1.

    estimator = RegularizedDictionary(embedding,
                                      likelihood,
                                      regularizer,
                                      #constraints=constraint,
                                      inference_type="LR",
                                      #use_constraint=False,
                                      check=checktype,
                                      custom_check = check_fn,
                                      accuracy= 1e-6,
                                      bound=bound,
                                      verbose=False)

    estimator.load_data((x[0, :].view(1, -1), y[0, :].view(1, 1)))
    n = x.size()[0]
    for i in range(1,n):
        estimator.add_points((x[i,:].view(1,-1), y[i,:].view(1,1)))
        estimator.fit()
        # lower = estimator.lcb(test)
        # upper = estimator.ucb(test)

    print (estimator.x)
    print (estimator.y)

    lower = estimator.lcb(test)
    upper = estimator.ucb(test)

    print ("====")
    bounds = upper - lower
    print(lower.T)
    print ("theta",theta.T)
    print (upper.T)
    print ("----------")
    bound1.append(bounds[0])
    bound2.append(bounds[1])

plt.plot(weightings,bound1,'o-', label = '$\\theta_1$', lw = 3)
plt.plot(weightings,bound2,'o-', label = '$\\theta_2$', lw = 3)
#plt.semilogx(lambdas, bound3,'o-', label = '$(\\theta_1 +\\theta_2)$', lw = 3)
plt.xlabel("diag$(\\lambda, 1)$")
plt.legend()
plt.show()



