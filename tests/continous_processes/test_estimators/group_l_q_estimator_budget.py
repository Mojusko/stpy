import torch
import numpy as np
import matplotlib.pyplot as plt
from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, ConcatEmbedding, MaskedEmbedding
from stpy.embeddings.bump_bases import FaberSchauderEmbedding, TriangleEmbedding
from stpy.embeddings.weighted_embedding import WeightedEmbedding
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.probability.regularizer import L2Regularizer, L1Regularizer, GroupL1L2Regularizer, NonConvexLqRegularizer, GroupNonCovexLqRegularizer
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.helpers.helper import interval_torch
from stpy.helpers.constraints import QuadraticInequalityConstraint, AbsoluteValueConstraint
from stpy.kernels import KernelFunction
from scipy.ndimage import gaussian_filter

"""
This script test and compares Lq estimators 
 compare L1, L2 and Lq estimators
"""

m = 64
d = 1
sigma = 0.01
lam = 1.
n = 64
N = 10

kernel_object = KernelFunction(gamma = 0.05, d = 1)
#embedding = HermiteEmbedding(m = m, d = 1)
xtest = interval_torch(n = n,d = 1)

embedding1 = NystromFeatures(kernel_object = kernel_object, m = m )
embedding1.fit_gp(xtest/2-0.5,None)
embedding2 = NystromFeatures(kernel_object = kernel_object, m = m )
embedding2.fit_gp(xtest/2+0.5,None)
embedding = ConcatEmbedding([embedding1,embedding2])

qs = [0.01]
groups = [list(range(m)), list(range(m, 2 * m, 1))]
print (groups)

regularizers = []
#regularizers += [L1Regularizer(lam = lam), L2Regularizer(lam = lam)]
#regularizers += [NonConvexLqRegularizer(lam = lam, q = q) for q in qs]
regularizers += [GroupNonCovexLqRegularizer(lam = lam, q = q, groups=groups) for q in qs]

likelihood = GaussianLikelihood(sigma=sigma)
names = []
#names += ["L1", "L2"]
#   names += ["L"+str(q) for q in qs]
names += ["group L"+str(q) for q in qs]

f = lambda x: torch.sin(x*20)
Xtrain = interval_torch(n = N, d= 1)
ytrain = f(Xtrain)

for name,regularizer in zip(names,regularizers):
    estimator = RegularizedDictionary(embedding, likelihood, regularizer)
    estimator.load_data((Xtrain,ytrain))
    estimator.fit()
    mean = estimator.mean(xtest)

    p = plt.plot(xtest, mean, label=name, lw=3, alpha=0.5)

    ucb = estimator.ucb(xtest, type="LR_static")
    lcb = estimator.lcb(xtest, type="LR_static")
    plt.fill_between(xtest.view(-1), lcb.view(-1), ucb.view(-1), alpha=0.1, color=p[0].get_color())

    print(name, "support:", torch.sum(estimator.theta_fit > 1e-8))


plt.plot(Xtrain,ytrain,'ko', lw = 3)
plt.plot(xtest,f(xtest),'k--', lw = 3)
plt.legend()
plt.show()