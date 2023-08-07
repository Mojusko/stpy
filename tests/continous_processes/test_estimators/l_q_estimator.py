import torch
import matplotlib.pyplot as plt
from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.regularization.regularizer import L2Regularizer, L1Regularizer
from stpy.helpers.helper import interval_torch
from stpy.kernels import KernelFunction
from stpy.regularization.constraints import NonConvexNormConstraint

"""
This script test and compares Lq estimators 
 compare L1, L2 and Lq estimators
"""

m = 64
d = 1
sigma = 0.01
lam = 1.
n = 4
N = 3
total_norm = 1.
xtest = interval_torch(n = n,d = 1)
kernel_object = KernelFunction(gamma = 0.05, d = 1)
embedding = HermiteEmbedding(m = m, d = 1)

total_norms = [1]
for pos,total_norm in enumerate(total_norms):
    lasso_regularizer = L1Regularizer(lam = lam)
    l2_regularizer = L2Regularizer(lam = lam)
    qs = [0.1]

    regularizers = [l2_regularizer,l2_regularizer]
    #regularizers +=  [NonConvexLqRegularizer(lam = lam, q = q) for q in qs]
    constraints = [lasso_regularizer.get_constraint_object(total_norm), l2_regularizer.get_constraint_object(total_norm)]
    #constraints=+ [None for q in qs]


    constraints += [ NonConvexNormConstraint(0.5, total_norm, m)]
    regularizers += [L2Regularizer(lam = lam)]

    likelihood = GaussianLikelihood(sigma=sigma)
    names = ["L1", "L2"]
    #names +=  ["L"+str(q) for q in qs]
    names += ["Lspecial"]

    f = lambda x: torch.sin(x*20)
    Xtrain = interval_torch(n = N, d= 1)
    ytrain = f(Xtrain)
    linestyles = ['-.','-','--']
    #plt.subplot(2,len(total_norms)//2,pos+1)
    for name,regularizer,constraint, linestyle  in zip(names,regularizers,constraints,linestyles):
        print (name)
        estimator = RegularizedDictionary(embedding, likelihood, regularizer, constraints=constraint, use_constraint=True)
        estimator.load_data((Xtrain,ytrain))
        estimator.fit()
        mean = estimator.mean(xtest)
        lcb = estimator.lcb(xtest)
        ucb = estimator.ucb(xtest)
        p = plt.plot(xtest, mean, label=name, linestyle = linestyle)
        plt.fill_between(xtest.view(-1),lcb.view(-1),ucb.view(-1), alpha = 0.1, color = p[0].get_color())
        print(name, "support:", torch.sum(estimator.theta_fit > 0.01))
        print (estimator.theta_fit.T)


    plt.plot(Xtrain,ytrain,'o')
    plt.plot(xtest,f(xtest),'k--')
    plt.legend()
plt.show()