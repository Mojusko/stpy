import cvxpy as cp
import mosek

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

class GroupedRegularizedDictionary(RegularizedDictionary):

    def description(self):
        return "grouped regularized dictionary object"


    def combine_groups_cvxpy(self, thetas,):
        theta_whole = cp.Variable((m,1))
        for theta,group in zip(self.groups,thetas):
            theta_whole += theta
        return


    def calculate(self):

        if self.fitted:
            pass
        else:
            # print ("Calculating.")
            ## cvxpy way
            thetas = []
            cs = []
            print ("Number of groups:",len(self.groups))
            print (self.m)

            for group in self.groups:
                thetas.append(cp.Variable((self.m, 1)))
                c = np.zeros(shape = (self.m,1))
                c[group,0] = 1.
                cs.append(c)
                print (c)

            w = cp.Variable((self.m, 1))
            zs = [cp.multiply(c,theta) for theta,c in zip(thetas,cs)]
            print (zs)
            z = None
            for e in zs:
                if z is None:
                    z = e
                else:
                    z = z+e
            print (z)
            likelihood = self.likelihood.get_objective_cvxpy()
            regularizer = self.regularizer.get_regularizer_cvxpy()


            objective = likelihood(w) + regularizer(w)

            constraints = [w==z, zs[0]+zs[1]==zs[2]]
            # if self.constraints is not None and self.use_constraint:
            #     set = self.constraints.get_cvxpy_constraint(thetas)
            #     constraints += set

            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.MOSEK, mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                                                      mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
                                                      mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
                                                      mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})
            self.thetas_fit = [torch.from_numpy(theta.value) for theta in thetas]
            self.theta_fit = sum(self.thetas_fit)
            self.fitted = True

if __name__ == "__main__":
    sigma = 0.005
    lam = 1.
    d = 1
    n = 16
    n = 512
    m = 128
    kernel_object = KernelFunction(gamma=0.1, d=1)

    embedding1 = TriangleEmbedding(m=m, d=1, kernel_object=kernel_object, interval=[-1, 0], offset=0.0)
    embedding2 = TriangleEmbedding(m=m, d=1, kernel_object=kernel_object, interval=[0, 1], offset=0.0)
    embedding = ConcatEmbedding([embedding1,embedding2])

    likelihood_base = GaussianLikelihood(sigma = sigma)

    N = 20
    torch.manual_seed(2)


    def zeroing(X):
        Y = X.clone()
        Y[X < 0.] = 0.
        return Y


    F = lambda X: (np.cos(X * 10.) + np.sin(X * 10.)) * zeroing(X)

    Xtrain = torch.rand(size=(10, d)).double() / 2
    ytrain = F(Xtrain) + sigma * torch.randn(size=(Xtrain.size()[0], 1))



    xtest = interval_torch(n = n,d = 1)
    groups = [list(range(m)), list(range(m, 2 * m, 1))]
    new_groups = groups.copy()
    alpha = 0.1
    weights = [alpha**2 for g in groups]

    # increase the combination of groups
    for j in range(len(groups)):
        for i in range(j+1,len(groups),1):
            new_groups.append(groups[j] + groups[i])
            weights.append(1.)

    regularizer = NestedGroupL1L2Regularizer(lam = lam, groups = new_groups, weights = weights)
    likelihood = GaussianLikelihood(sigma=sigma)
    estimator_train = GroupedRegularizedDictionary(embedding, likelihood, regularizer, groups=new_groups)

    estimator_train.load_data((Xtrain,ytrain))
    estimator_train.fit()
    mean = estimator_train.mean(xtest)

    plt.plot(Xtrain, ytrain, 'ro', ms=15)
    plt.plot(xtest, F(xtest), lw = 4)
    p = plt.plot(xtest, mean, lw = 4, label = "$\\lambda = "+str(lam)+", \\alpha ="+str(alpha)+" $")
    plt.show()
