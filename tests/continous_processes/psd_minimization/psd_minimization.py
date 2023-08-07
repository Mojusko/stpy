from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, ConcatEmbedding, MaskedEmbedding
import pymanopt
import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, ConcatEmbedding, MaskedEmbedding
from stpy.kernels import KernelFunction
from stpy.helpers.helper import interval, interval_torch
from stpy.probability.gaussian_likelihood import GaussianLikelihood
import matplotlib.pyplot as plt
from stpy.continuous_processes.regularized_dictionary import RegularizedDictionary
from stpy.continuous_processes.nystrom_fea import NystromFeatures

if __name__ == "__main__":


    N = 10
    n = 256
    d = 1
    eps = 0.01
    s = 0.01
    m = 128

    V = torch.linalg.qr(torch.randn(size=(m, m)).double())[0]

    f = lambda x: torch.sin(x * 20) * (x > 0).double()
    Xtrain = interval_torch(n=N, d=1)
    ytrain = f(Xtrain)

    xtest = torch.from_numpy(interval(n, d, L_infinity_ball=1))
    kernel_object = KernelFunction(gamma=0.05, d=1)

    #embedding = HermiteEmbedding(m=m, gamma = 1.)

    embedding1 = NystromFeatures(kernel_object=kernel_object, m=m//2)
    embedding1.fit_gp(xtest / 2 - 0.5, None)
    embedding2 = NystromFeatures(kernel_object=kernel_object, m=m//2)
    embedding2.fit_gp(xtest / 2 + 0.5, None)
    embedding = ConcatEmbedding([embedding1, embedding2])


    theta = cp.Variable((m,1))
    A1 = cp.Variable((m//2,m//2), PSD = True)
    A2 = cp.Variable((m//2, m//2), PSD=True)
    A3 = cp.Variable((m//2, m//2))
    t = cp.Variable()

    likelihood = GaussianLikelihood(sigma=s)
    estimator = RegularizedDictionary(embedding, likelihood)
    data = (embedding.embed(Xtrain),ytrain)
    estimator.load_data(data)
    likelihood = estimator.likelihood
    likelihood.load_data(data)

    total_trace = 5.

    objective = likelihood.get_objective_cvxpy()(theta)
    A = cp.bmat([[A1,A3],[A3,A2]])
    constraints = [cp.matrix_frac(theta, A) <= 1,  cp.trace(A) <= total_trace, A >> 0]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver = cp.MOSEK, verbose = True)

    estimator.theta_fit = theta.value
    estimator.fitted = True
    print (prob.value)
    #plt.imshow(A.value)
    #plt.show()
    if theta.value is not None:
        mu = estimator.mean(xtest)
        plt.plot(xtest,mu, 'b', lw = 3, label = 'opt')

    plt.plot(Xtrain,ytrain,'ko', lw = 3)
    plt.plot(xtest,f(xtest),'k--', lw = 3)


    theta = cp.Variable((m,1))
    V = cp.Parameter((m, m))
    objective = likelihood.get_objective_cvxpy()(theta)
    a = cp.Variable(m)
    A = cp.Variable((m,m))

    constraints = [cp.matrix_frac(V.T@theta, cp.diag(a)) <= 1., a>=0, cp.sum(a)<=total_trace]
    prob = cp.Problem(cp.Minimize(objective), constraints)



    manifold = pymanopt.manifolds.Stiefel(m,m)

    def opt(V_val):
        V.value = V_val
        prob.solve(solver = cp.MOSEK, verbose = False)
        return theta.value

    @pymanopt.function.numpy(manifold)
    def cost(V_val):
        V.value = V_val
        prob.solve(requires_grad=True, solver = cp.SCS)
        return prob.value

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(V_val):
        V.value = V_val
        prob.solve(requires_grad=True)
        prob.backward()
        return V.gradient

    print ("INITIAL COST:", cost(np.eye(m)))
    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
    optimizer = pymanopt.optimizers.SteepestDescent(min_step_size=1e-15)
    result = optimizer.run(problem, initial_point = np.eye(m))
    V_val = result.point
    #V_val = np.eye(m)
    #print (result)
    print (V_val@V_val.T)
    print ("END COST:", cost(V_val))

    estimator.theta_fit = opt(V_val)
    estimator.fitted = True

    mu = estimator.mean(xtest)
    plt.plot(xtest,mu, 'r--', lw = 3, label = 'ortho opt')







    estimator.theta_fit = opt(np.eye(m))
    mu = estimator.mean(xtest)
    plt.plot(xtest,mu, 'g--', lw = 3, label = 'A identity')


    # simplified objective
    theta = cp.Variable((m,1))
    objective = likelihood.get_objective_cvxpy()(theta)
    constraints = [cp.sum_squares(theta) <= total_trace/m]
    prob_simple = cp.Problem(cp.Minimize(objective), constraints)
    prob_simple.solve()
    print ("SIMPLE COST:",prob_simple.value)
    estimator.theta_fit = theta.value
    mu = estimator.mean(xtest)
    plt.plot(xtest,mu, 'tab:purple', lw = 3, label = 'simple solution')

    theta = cp.Variable((m,1))
    V = cp.Parameter((m, m))
    objective = likelihood.get_objective_cvxpy()(theta)
    a = cp.Variable(m)
    A = cp.Variable((m,m), PSD=True)
    constraints = [cp.matrix_frac(theta, cp.diag(a)) <= 1., a>=0, cp.sum(a)<=total_trace]
    prob_complicated = cp.Problem(cp.Minimize(objective), constraints)
    prob_complicated.solve(solver = cp.MOSEK , verbose = True)
    estimator.theta_fit = theta.value
    mu = estimator.mean(xtest)

    plt.plot(xtest,mu, 'tab:brown', lw = 3, label = 'soln')

    plt.legend()
    plt.show()