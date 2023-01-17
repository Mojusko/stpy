import torch
import numpy as np
import matplotlib.pyplot as plt
from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, ConcatEmbedding, MaskedEmbedding
from stpy.embeddings.bump_bases import FaberSchauderEmbedding, TriangleEmbedding
from stpy.embeddings.weighted_embedding import WeightedEmbedding
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.probability.regularizer import L2Regularizer, L1Regularizer, GroupL1L2Regularizer, NonConvexLqRegularizer
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.helpers.helper import interval_torch
from stpy.helpers.constraints import QuadraticInequalityConstraint, AbsoluteValueConstraint
from stpy.kernels import KernelFunction
from scipy.ndimage import gaussian_filter

"""
This script test and compares Lq estimators 
 compare L1, L2 and Lq estimators
"""

m = 128
d = 1
sigma = 0.01
lam = 1.
n = 256
N = 10

kernel_object = KernelFunction(gamma = 0.05, d = 1)
embedding = HermiteEmbedding(m = m, d = 1)



xtest = interval_torch(n = n,d = 1)

qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
regularizers = [L1Regularizer(lam = lam), L2Regularizer(lam = lam)]
regularizers = regularizers + [NonConvexLqRegularizer(lam = lam, q = q) for q in qs]
likelihood = GaussianLikelihood(sigma=sigma)
names = ["L1", "L2"] + ["L"+str(q) for q in qs] 

f = lambda x: torch.sin(x*20)
Xtrain = interval_torch(n = N, d= 1)
ytrain = f(Xtrain)

for name,regularizer in zip(names,regularizers):
    estimator = RegularizedDictionary(embedding, likelihood, regularizer)
    estimator.load_data((Xtrain,ytrain))
    estimator.fit()
    mean = estimator.mean(xtest)
    print(name, "support:", torch.sum(estimator.theta_fit > 1e-8))
    plt.plot(xtest, mean, label = name)

plt.plot(Xtrain,ytrain,'o')
plt.plot(xtest,f(xtest),'k--')
plt.legend()
plt.show()