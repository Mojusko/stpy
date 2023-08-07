import torch
import numpy as np
import matplotlib.pyplot as plt

from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.bump_bases import FaberSchauderEmbedding
from stpy.embeddings.weighted_embedding import WeightedEmbedding
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.regularization.regularizer import L2Regularizer, L1Regularizer
from stpy.helpers.helper import interval_torch
from stpy.regularization.constraints import QuadraticInequalityConstraint, AbsoluteValueConstraint
from stpy.kernels import KernelFunction

m = 64
d = 1
sigma = 0.1
lam = 1.
n = 256

I = torch.eye(m).double()
budget = m*10e10
kernel_object = KernelFunction(gamma = 0.1, d = 1)
#embedding = TriangleEmbedding(m = m, d = 1, B = 10, b = -10, kernel_object=kernel_object)

embedding_base = FaberSchauderEmbedding(m = m, d = 1, kernel_object=None, offset=0)
# this defines the decay of the functions
def decay_function(emb):
    return (emb.hierarchical_mask()+1)**(-15)

print (decay_function(embedding_base))

embedding = WeightedEmbedding(embedding_base,weight_function=decay_function)

# embedding = RFFEmbeddQing(m = m, d=1, gamma = 0.1)

likelihood = GaussianLikelihood(sigma = sigma)
regularizer_L2 = L2Regularizer(lam = lam)
regularizer_L1 = L1Regularizer(lam = lam)


constraint_L2 = QuadraticInequalityConstraint(Q = I, c = budget)
constraint_L1 = AbsoluteValueConstraint(c = np.sqrt(budget))

estimator_L2_L2 = RegularizedDictionary(embedding, likelihood, regularizer_L2,
                                        constraints = constraint_L2, use_constraint=False)
estimator_L1_L2 = RegularizedDictionary(embedding, likelihood, regularizer_L1,
                                        constraints = constraint_L2, use_constraint=False)
estimator_L2_L1 = RegularizedDictionary(embedding, likelihood, regularizer_L2,
                                        constraints = constraint_L1, use_constraint=False)
estimator_L1_L1 = RegularizedDictionary(embedding, likelihood, regularizer_L1,
                                        constraints = constraint_L1, use_constraint=False)

estimators = [estimator_L2_L2,estimator_L2_L1,estimator_L1_L2,estimator_L1_L1]
names = ["reg:L2 con:L2", "reg:L2 con:L1", "reg:L1 con:L2", "reg:L1 con:L1"]
styles = ["-","--","-","--"]
N = 1
v = torch.randn(size = (m,1)).double()
F = lambda X: embedding.embed(X)@v
X = torch.Tensor([[0.5]]).double()
y = F(X)
xtest = interval_torch(n = n,d = 1)

plt.plot(xtest, F(xtest), lw = 5)
plt.plot(X, y, 'ro', ms = 25)

for j,estimator in enumerate(estimators):
    print ("Calculating:",names[j])
    estimator.load_data((X,y))
    estimator.fit()
    mean = estimator.mean(xtest)

    #ucb = estimator.ucb(xtest, type = "LR_static")
    #lcb = estimator.lcb(xtest, type = "LR_static")

    #plt.title("Norm: "+str(torch.norm(estimator.theta_fit)**2))
    plt.plot(xtest, mean, label = names[j], lw = 4, linestyle = styles[j])
    #plt.fill_between(xtest.view(-1), lcb.view(-1), ucb.view(-1), alpha = 0.1)

plt.legend(fontsize = 35)
plt.show()
