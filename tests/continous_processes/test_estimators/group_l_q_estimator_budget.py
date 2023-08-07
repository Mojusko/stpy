import torch
import matplotlib.pyplot as plt
from stpy.embeddings.embedding import ConcatEmbedding
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.helpers.helper import interval_torch
from stpy.kernels import KernelFunction
from stpy.regularization.simplex_regularizer import SupRegularizer
from stpy.continuous_processes.mkl_estimator import MultipleKernelLearner
"""
This script test and compares Lq estimators 
 compare L1, L2 and Lq estimators
"""

m = 128
d = 1
sigma = 0.01
lam = 1.
n = 128
N = 10

kernel_object = KernelFunction(gamma = 0.05, d = 1)
#embedding = HermiteEmbedding(m = m, d = 1)
xtest = interval_torch(n = n,d = 1)

embedding1 = NystromFeatures(kernel_object = kernel_object, m = m )
embedding1.fit_gp(xtest/2-0.7,None)
embedding2 = NystromFeatures(kernel_object = kernel_object, m = m )
embedding2.fit_gp(xtest/2+0.7,None)
embedding = ConcatEmbedding([embedding1,embedding2])

def k1(x,y,**kwagrs):
    return (embedding1.embed(x)@embedding1.embed(y).T).T

def k2(x,y,**kwagrs):
    return (embedding2.embed(x)@embedding2.embed(y).T).T

kernel_object_1 = KernelFunction(kernel_function = k1)
kernel_object_2 = KernelFunction(kernel_function = k2)

kernels = [kernel_object_1, kernel_object_2]
regularizer = SupRegularizer(d=len(kernels), lam=0.99, constrained=True)
mkl = MultipleKernelLearner(kernels, regularizer=regularizer)

f = lambda x: torch.sin(x*20)*(x<0).double() + (1e-5)*torch.sin(x*20)*(x>0).double()
Xtrain = interval_torch(n = N, d= 1, L_infinity_ball=0.25) - 0.75
ytrain = f(Xtrain)

#
# qs = [0.1]
# groups = [list(range(m)), list(range(m, 2 * m, 1))]
# print (groups)
#
# total_norms = [[20,20,10,10,10]]
# for pos,total_norm in enumerate(total_norms):
#     lasso_regularizer = L1Regularizer(lam = lam)
#     l2_regularizer = L2Regularizer(lam = lam)
#
#     regularizers = []
#     regularizers += [l2_regularizer, l2_regularizer]
#     regularizers += [ l2_regularizer for _ in qs]
#     constraints = [lasso_regularizer.get_constraint_object(total_norm[0]), l2_regularizer.get_constraint_object(total_norm[1])]
#     constraints += [ NonConvexGroupNormConstraint(q, total_norm[2+index], m, groups) for index,q in enumerate(qs)]
#
#     likelihood = GaussianLikelihood(sigma=sigma)
#     names = []
#     names += ["L1", "L2"]
#     #   names += ["L"+str(q) for q in qs]
#     names += ["group "+str(q) for q in qs]
#

#     linestyles = ['-.','-','--',"--","--"]
#     colorstyles = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:brown']
#     for name, regularizer, constraint, linestyle, color in zip(names, regularizers, constraints, linestyles, colorstyles):
#         estimator = RegularizedDictionary(embedding, likelihood, regularizer, constraints=constraint, use_constraint=True)
#         estimator.load_data((Xtrain,ytrain))
#         estimator.fit()
#         mean = estimator.mean(xtest)
#         lcb = estimator.lcb(xtest)
#         ucb = estimator.ucb(xtest)
#         p = plt.plot(xtest, mean, label=name, linestyle = linestyle, lw = 3, color = color )
#         plt.fill_between(xtest.view(-1),lcb.view(-1),ucb.view(-1), alpha = 0.2, color = p[0].get_color())
#         print(name, "support:", torch.sum(estimator.theta_fit > 0.01))
#         print (estimator.theta_fit.T)

mkl.load_data((Xtrain, ytrain))
mkl.fit()
mean = mkl.mean(xtest)
p = plt.plot(xtest, mean, label="MKL", linestyle="-", lw=3, color='tab:purple')

plt.plot(Xtrain,ytrain,'ko', lw = 3)
plt.plot(xtest,f(xtest),'k--', lw = 3)
plt.legend()
plt.show()