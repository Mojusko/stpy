import pymanopt
import cvxpy as cp
import numpy as np
import torch


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
    m = 32

    def stable_rank(A):
        return np.trace(A)/np.max(np.linalg.eigh(A)[0])


    V = torch.linalg.qr(torch.randn(size=(m, m)).double())[0]

    f = lambda x: 0.5*torch.sin(x * 20) * (x > 0).double() + 0.5*torch.sin(x * 30) * (x > 0).double()
    Xtrain = interval_torch(n=N, d=1)
    ytrain = f(Xtrain)

    xtest = torch.from_numpy(interval(n, d, L_infinity_ball=1))
    kernel_object = KernelFunction(gamma=0.05, d=1)

    # embedding = HermiteEmbedding(m=m, gamma = 1.)

    embedding1 = NystromFeatures(kernel_object=kernel_object, m=m // 2)
    embedding1.fit_gp(xtest / 2 - 0.5, None)
    embedding2 = NystromFeatures(kernel_object=kernel_object, m=m // 2)
    embedding2.fit_gp(xtest / 2 + 0.5, None)
    embedding = ConcatEmbedding([embedding1, embedding2])

    theta = cp.Variable((m, 1))
    A1 = cp.Variable((m // 2, m // 2), PSD=True)
    A2 = cp.Variable((m // 2, m // 2), PSD=True)
    A3 = cp.Variable((m // 2, m // 2))
    l = cp.Variable((1,1))
    s = cp.Parameter((1, 1), nonneg = True)

    likelihood = GaussianLikelihood(sigma=s)
    estimator = RegularizedDictionary(embedding, likelihood)
    data = (embedding.embed(Xtrain), ytrain)
    estimator.load_data(data)
    likelihood = estimator.likelihood
    likelihood.load_data(data)

    total_trace = 2.
    objective = likelihood.get_objective_cvxpy()(theta)
    A = cp.bmat([[A1, A3], [A3, A2]])
    s.value = np.array([[1.]])
    constraints = [cp.matrix_frac(theta, A) <= 1, cp.trace(A) <= total_trace*l, A >> 0,cp.lambda_max(A)<=l]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)

    estimator.theta_fit = theta.value
    estimator.fitted = True
    print (prob.value)
    print (np.max(np.linalg.eigh(A.value)[0]))
    print (l.value)
    print("--------------")

    if theta.value is not None:
        mu = estimator.mean(xtest)
        plt.plot(xtest,mu, 'b', lw = 3, label = 'opt')

    plt.plot(Xtrain,ytrain,'ko', lw = 3)
    plt.plot(xtest,f(xtest),'k--', lw = 3)

    constraints = [cp.matrix_frac(theta, A) <= 1, cp.trace(A) <= total_trace*l, A >> 0,cp.lambda_max(A)<=l, l<=s]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)

    def cost(z):
        s.value = z
        prob.solve()
        return prob.value, total_trace * l.value, l.value,  (np.max(np.linalg.eigh(A.value)[0])), np.trace(A.value), stable_rank(A.value)

    z_vals = np.logspace(-5,5,20, base = 2)
    l_vals = []
    eigvals = []
    differences = []
    for z in z_vals:
        prob_val, _, l_val, eigv, _ , _  = cost(np.array([[z]]))
        estimator.theta_fit = theta.value
        estimator.fitted = True
        mu = estimator.mean(xtest)
        l_vals.append(float(l_val))
        eigvals.append(float(eigv))
        differences.append(float(l_val) - float(eigv))

        print (z, float(l_val) - float(eigv))

        if float(l_val) - float(eigv) <= 1e-2 and float(l_val) - float(eigv)>=0:
            plt.plot(xtest,mu, 'g--', lw = 3, label = 'stable-rank')
    plt.show()

    plt.plot(z_vals.reshape(-1),l_vals, label = 'lvals')
    plt.plot(z_vals.reshape(-1),eigvals, label = 'eig')
    # plt.plot(z_vals.reshape(-1), differences, label='diff')
    plt.legend()
    plt.show()


    #
    # # Fix an eigenvector
    # v_init = np.zeros(shape=(m, 1))
    # v_init[0, 0] = 1.
    #
    # l = cp.Variable((1,1))
    # A = cp.bmat([[A1, A3], [A3, A2]])
    # v = cp.Parameter((m,1))
    # v.value = v_init
    #
    # # this makes sure that l is the largest eigenvalue of A,
    # constraints = [cp.matrix_frac(theta, A) <= 1, cp.trace(A) <= total_trace*l, A >> 0, A@v == l *v, cp.lambda_max(A)<=l]
    # prob = cp.Problem(cp.Minimize(objective), constraints)
    # prob.solve(solver=cp.MOSEK, verbose=True)
    #
    # estimator.theta_fit = theta.value
    # estimator.fitted = True
    #
    # mu = estimator.mean(xtest)
    # print (np.max(np.linalg.eigh(A.value)[0]))
    # print (l.value)
    # print("--------------")
    #
    # plt.plot(xtest, mu, 'r', lw=3, label='opt-2')
    #
    #
    # def value(w):
    #     v.value = w
    #     prob.solve(requires_grad=True, solver = cp.SCS)
    #     leig = np.max(np.linalg.eigh(A.value)[0])
    #     lval = l.value
    #     return prob.value, lval, leig
    #
    # for j in range(m):
    #     w = np.zeros(shape = (m,1))
    #     w[j,0] =1
    #     print (j, value(w))
    #     estimator.theta_fit = theta.value
    #     estimator.fitted = True
    #     mu = estimator.mean(xtest)
    #
    #     plt.plot(xtest, mu, 'g--', lw=1, label='opt-2')
    # plt.show()
    #
    # #
    # # def euclidean_gradient(w):
    # #     v.value = w
    # #     prob.solve(requires_grad=True, solver = cp.SCS)
    # #     prob.backward()
    # #     return v.gradient
    # #
    # # def proj(w):
    # #     return w/np.linalg.norm(w)
    # #
    # # steps = 100
    # # eta = 1e-8
    # # w = v_init
    # # for i in range(steps):
    # #     grad = euclidean_gradient(w)
    # #     w = w - eta * grad
    # #     w = proj(w)
    # #     print (i, value(w))