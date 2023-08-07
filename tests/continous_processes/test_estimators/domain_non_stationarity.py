import torch
import numpy as np
import matplotlib.pyplot as plt

from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.embedding import ConcatEmbedding
from stpy.embeddings.bump_bases import TriangleEmbedding
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.regularization.regularizer import NestedGroupL1L2Regularizer
from stpy.helpers.helper import interval_torch
from stpy.kernels import KernelFunction

m = 200
d = 1
sigma = 0.005
lam = 1.
n = 256

I = torch.eye(m).double()
budget = m*1
kernel_object = KernelFunction(gamma = 0.1, d = 1)

embedding1 = TriangleEmbedding(m = m, d = 1, kernel_object=kernel_object, interval=[-1,0], offset=0.0)
embedding2 = TriangleEmbedding(m = m, d = 1, kernel_object=kernel_object, interval=[0,1], offset=0.0)

embedding = ConcatEmbedding([embedding1,embedding2])

likelihood_base = GaussianLikelihood(sigma = sigma)


# for w,g in zip(weights,new_groups):
#     print (w,g)

# regularizer_base = NestedGroupL1L2Regularizer(lam=1., groups=new_groups,weights=weights )
# constraint_base = regularizer_base.get_constraint_object(budget)
# estimator = RegularizedDictionary(embedding, likelihood_base, regularizer_base, constraints=constraint_base, use_constraint=False)

N = 20
torch.manual_seed(2)

def zeroing(X):
    Y = X.clone()
    Y[ X < 0.] = 0.
    return Y

F = lambda X: (np.cos(X*10.)+np.sin(X*10.))*zeroing(X)
# X = torch.rand(size = (N,d)).double()*0.25+0.5
# y = F(X)
#
# Xpoint = torch.Tensor([[0.],[0.5]]).double()
# ypoint = torch.Tensor([[0.],[0.]]).double()
#
# X = torch.vstack([X,Xpoint])
# y = torch.vstack([y,ypoint])
# estimator.load_data((X,y))
# estimator.fit()
#
# F = lambda X: estimator.mean(X)


Xtrain = torch.rand(size=(10, d)).double()/2
ytrain = F(Xtrain) + sigma * torch.randn(size=(Xtrain.size()[0], 1))



def update():
    pass
alphas = [5,10]#,0.01,0.001]
lams_uns = [0.01,0.05,0.1]
# alphas = [0.01]
# lams_uns = [0.1]

fig, axs = plt.subplots(len(alphas), len(lams_uns))

for index1, alpha in enumerate(alphas):
    lams = [la/alpha for la in lams_uns]#, 0.01/alpha]#,16.,32.,64.,128.]

    for index2, lam in enumerate(lams):
        print ("Regularizer:", alpha, lam)

        xtest = interval_torch(n = n,d = 1)
        groups = [list(range(m)), list(range(m, 2 * m, 1))]
        new_groups = groups.copy()
        weights = [alpha**2 for g in groups]
        for j in range(len(groups)):
            for i in range(j + 1, len(groups), 1):
                new_groups.append(groups[j] + groups[i])
                weights.append(1.)

        regularizer = NestedGroupL1L2Regularizer(lam = lam, groups = new_groups, weights = weights)
        constraint = regularizer.get_constraint_object(budget)
        likelihood = GaussianLikelihood(sigma=sigma)
        estimator_train = RegularizedDictionary(embedding, likelihood, regularizer, constraints = constraint, use_constraint=True)

        estimator_train.load_data((Xtrain,ytrain))
        estimator_train.fit()
        mean = estimator_train.mean(xtest)




        if max(len(alphas),len(lams_uns))>1:
            #axs[index1,index2].subplot(len(lams),len(alphas),index1+1, index2+1)
            axs[index1,index2].plot(Xtrain, ytrain, 'ro', ms=15)
            axs[index1,index2].plot(xtest, F(xtest), lw = 4)
            p = axs[index1,index2].plot(xtest, mean, lw = 4, label = "$\\lambda = "+str(lam)+", \\alpha ="+str(alpha)+" $")

            # xtest1 = torch.linspace(0.0,0.5,n//4).double().view(-1,1)
            # xtest2 = torch.linspace(-1.0,-0.5,n//4).double().view(-1,1)
            # conf_xtest = torch.vstack([xtest1,xtest2])
            ucb = estimator_train.ucb(xtest, type = "LR_static")
            lcb = estimator_train.lcb(xtest, type = "LR_static")
            axs[index1,index2].fill_between(xtest.view(-1), lcb.view(-1), ucb.view(-1), alpha = 0.1, color = p[0].get_color())
            #axs[index1,index2].legend(fontsize = 15)
        else:
            axs.plot(Xtrain, ytrain, 'ro', ms=15)
            axs.plot(xtest, F(xtest), lw=4)
            p = axs.plot(xtest, mean, lw=4,
                                     label="$\\lambda = " + str(lam) + ", \\alpha =" + str(alpha) + " $")

            # xtest1 = torch.linspace(0.0,0.5,n//4).double().view(-1,1)
            # xtest2 = torch.linspace(-1.0,-0.5,n//4).double().view(-1,1)
            # conf_xtest = torch.vstack([xtest1,xtest2])
            ucb = estimator_train.ucb(xtest, type="LR_static")
            lcb = estimator_train.lcb(xtest, type="LR_static")
            axs.fill_between(xtest.view(-1), lcb.view(-1), ucb.view(-1), alpha=0.1,
                                             color=p[0].get_color())
            #axs.legend(fontsize=15)
plt.savefig("image.png", dpi = 300)
plt.show()