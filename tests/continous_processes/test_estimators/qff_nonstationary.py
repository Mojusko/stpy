import torch
import numpy as np
import matplotlib.pyplot as plt
from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.embedding import HermiteEmbedding, ConcatEmbedding
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.regularization.regularizer import GroupL1L2Regularizer
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.helpers.helper import interval_torch
from stpy.kernels import KernelFunction

m = 128
d = 1
sigma = 0.01
lam = 1.
n = 256
n_small = 16


I = torch.eye(m).double()
budget = 100
kernel_object = KernelFunction(gamma = 0.05, d = 1)

embedding_base = HermiteEmbedding(m = m, d = 1)

def zero_out_interval(x,interval):
    mask1 = x[:,0] > interval[0]
    mask2 = x[:,0] < interval[1]
    #return torch.from_numpy(gaussian_filter(torch.logical_and(mask1,mask2).double(),sigma=10))
    return torch.logical_and(mask1,mask2).double()


xtest = interval_torch(n = n,d = 1)


embedding1 = NystromFeatures(kernel_object = kernel_object, m = m )
embedding1.fit_gp((xtest-1)/2-0.5,None)
embedding2 = NystromFeatures(kernel_object = kernel_object, m = m )
embedding2.fit_gp((xtest-1)/2,None)
embedding3 = NystromFeatures(kernel_object = kernel_object, m = m )
embedding3.fit_gp((xtest+1)/2,None)
embedding4 = NystromFeatures(kernel_object = kernel_object, m = m )
embedding4.fit_gp((xtest+1)/2+0.5,None)

embedding = ConcatEmbedding([embedding1,embedding2,embedding3,embedding4])

likelihood_base = GaussianLikelihood(sigma = sigma)
groups = [list(range(m)),list(range(m,2*m,1)),list(range(2*m,3*m,1)),list(range(3*m,4*m,1))]

regularizer_base = GroupL1L2Regularizer(lam=1., groups=groups)
constraint_base = regularizer_base.get_constraint_object(budget)

estimator = RegularizedDictionary(embedding, likelihood_base, regularizer_base, constraints=constraint_base, use_constraint=False)

lams = [1.]#,16.,32.,64.,128.]
N = 3
v = torch.randn(size = (embedding.get_m(),1)).double()
for i in [0,1,3]:
    v[groups[i]] = 0.
v = (v/np.sqrt(regularizer_base.eval(v)))

F = lambda X: embedding.embed(X)@v*np.sqrt(budget)
X = torch.rand(size = (10,d)).double()*0.25+0.1
y = F(X)

#Xpoint = torch.Tensor([[0.],[0.5]]).double()
#ypoint = torch.Tensor([[0.],[0.]]).double()

#X = torch.vstack([X,Xpoint])
#y = torch.vstack([y,ypoint])
estimator.load_data((X,y))
estimator.fit()

F = lambda X: estimator.mean(X)
Xtrain = torch.rand(size=(N, d)).double() * 0.5
ytrain = F(Xtrain) + sigma * torch.randn(size=(Xtrain.size()[0], 1))

lams = [8.,16.,32.]#,16.,32.,64.,128.]
##lams = [1.,128.]
epsilon = 1e-1
#lams = [1.]
for index, lam in enumerate(lams):

    print (index,':',lam)
    print ("budget:",budget)

    plt.subplot(len(lams),1,index+1)
    plt.plot(Xtrain, ytrain, 'ro', ms=25)
    plt.ylim([-3,3])
    regularizer = GroupL1L2Regularizer(lam = lam, groups = groups)
    constraint = regularizer.get_constraint_object(budget)
    likelihood = GaussianLikelihood(sigma=sigma)
    estimator_train = RegularizedDictionary(embedding, likelihood, regularizer, constraints = constraint, use_constraint=True)


    xtest = interval_torch(n = n,d = 1)
    xtest_small = interval_torch(n = n_small, d = 1)
    plt.plot(xtest, F(xtest), lw = 5)

    estimator_train.load_data((Xtrain,ytrain))
    estimator_train.fit()

    mean = estimator_train.mean(xtest)
    mean_small = estimator_train.mean(xtest_small)

    print(regularizer.eval(v))
    print(regularizer.eval(estimator_train.theta_fit))
    print(regularizer_base.eval(estimator_train.theta_fit))

    p = plt.plot(xtest, mean, lw = 4, label = "$||f|| \leq "+str(budget/lam)+"$")
    #p2 = plt.plot(xtest_small, mean_small,'o-', ms = 25, lw = 4, label = "$||f|| \leq "+str(budget/lam)+"$")

    #
    ucb = estimator_train.ucb(xtest_small, type = "LR_static")
    lcb = estimator_train.lcb(xtest_small, type = "LR_static")
    #
    #plt.errorbar(xtest_small.view(-1), mean_small.view(-1),yerr = ucb.view(-1), ms = 25,alpha = 1., color = p[0].get_color(), lw=5)
    plt.fill_between(xtest_small.view(-1),lcb.view(-1), ucb.view(-1),alpha = 0.1, color = p[0].get_color())
    plt.plot(xtest, xtest*0 + epsilon, 'k--')
    plt.legend(fontsize = 35)
plt.show()
