import cvxpy as cp
import numpy as np
import torch
import matplotlib.pyplot as plt
from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding, ConcatEmbedding, MaskedEmbedding
from stpy.kernel import KernelFunction
from stpy.helpers.helper import interval, interval_torch
from stpy.regression.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.nystrom_fea import NystromFeatures
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.regression.regularized_dictionary_psd import RegularizedDictionaryPSD
from stpy.regularization.sdp_constraint import SDPConstraint

if __name__ == "__main__":


    N = 12
    n = 128
    d = 1
    eps = 0.01
    s = 0.1
    m = 80

    f = lambda x: 0.5 * torch.sin(x * 20) * (x > 0).double()+\
                  0.25 * torch.sin(x * 10) * (x > 0).double()+\
                  0.25 * torch.sin(x * 5) * (x > 0).double()

    #Xtrain = interval_torch(n=N, d=1)
    Xtrain = interval_torch(n=N, d=1, L_infinity_ball=0.4)+0.6
    ytrain = f(Xtrain)

    torch.manual_seed(22)
    np.random.seed(22)

    xtest = torch.from_numpy(interval(n, d, L_infinity_ball=1))
    kernel_object = KernelFunction(gamma=0.03, d=1)
    xtest_nystrom = torch.from_numpy(interval(m, d, L_infinity_ball=1))

    embedding = NystromFeatures(kernel_object=kernel_object, m = m)
    embedding.fit_gp(xtest_nystrom, None)

    likelihood = GaussianLikelihood(sigma=s)
    cmap = plt.get_cmap('viridis')
    num_colors = 1
    max_m = 32
    min_m = 2
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    pairs = list(zip(np.arange(min_m,max_m+1,max_m//num_colors), colors))



    for rank,color in pairs:

        regularization = SDPConstraint(type="stable-rank", rank=rank,
                                       lambda_max_constraint = 2.,
                                       pattern = "diagonal")

        regularization2 = SDPConstraint(type="lambda_max",
                                        lambda_max_constraint=2.
                                        ,pattern = "diagonal")

        embedding_taylored = NystromFeatures(kernel_object=kernel_object, m=N, approx = 'svd')
        embedding_taylored.fit_gp(Xtrain, None)

        estimator = RegularizedDictionaryPSD(embedding_taylored, likelihood,
                                             sdp_constraint=regularization,
                                             pattern="diagonal",
                                             norm_regularization=False)
        estimator.load_data((Xtrain, ytrain))
        estimator.fit()

        estimator_classical = RegularizedDictionaryPSD(embedding_taylored, likelihood,
                                                       sdp_constraint=regularization2,
                                                       pattern="diagonal",
                                                       norm_regularization=False)

        estimator_classical.load_data((Xtrain, ytrain))
        estimator_classical.fit()

        print ("Bias calculation.")
        std_det = torch.zeros(size=(n, 1)).double()
        std_det2 = torch.zeros(size=(n, 1)).double()

        # for i in range(n):
        #     std_det[i] = estimator.bias(xtest[i].view(1, -1))
        #     std_det2[i] = estimator.bias_enumerate_2(xtest[i].view(1,-1))
        #     print (std_det2[i])


        # STABLE = False
        # if STABLE:
        #     estimator_stable = RegularizedDictionaryPSD(embedding, likelihood,
        #                                                 sdp_constraint=regularization,
        #                                                 pattern="diagonal")
        #
        #     estimator_stable.load_data((Xtrain, ytrain))
        #     estimator_stable.fit()
        #
        #     #
        #     std_det_sup = torch.zeros(size=(n, 1)).double()
        #     for i in range(n):
        #         std_det_sup[i] = estimator_stable.bias(xtest[i].view(1, -1))
        #     mu_sup = estimator_stable.mean(xtest)
        #     print("ESTIMATOR STABLE")
        #     print(estimator_stable.l_fit)
        #     print(torch.max(torch.linalg.eigvalsh(estimator_stable.A_fit)))
        #     plt.plot(xtest, mu_sup, 'r--', lw=3, label='stable')
        #     plt.fill_between(xtest.view(-1), mu_sup.view(-1) - std_det_sup.view(-1), mu_sup.view(-1) + std_det_sup.view(-1), color='r',
        #                  alpha=.1)


        print("ESTIMATOR DIAGONAL")
        print("diag:",torch.diag(estimator.A_fit))
        print("trace:",torch.trace(estimator.A_fit))
        print("norm:",torch.linalg.norm(estimator.theta_fit))

        mu = estimator.mean(xtest)
        mu_classical = estimator_classical.mean(xtest)
        print("ESTIMATOR CLASSICAL")
        print("diag:",torch.diag(estimator_classical.A_fit))
        print("teace:",torch.trace(estimator_classical.A_fit))
        print("norm:",torch.linalg.norm(estimator_classical.theta_fit))

        # mu_classical2 = estimator_classical2.mean(xtest)
        # print("ESTIMATOR CLASSICAL 2")
        # print(torch.diag(estimator_classical2.A_fit))
        # print(torch.trace(estimator_classical2.A_fit))
        # print(torch.linalg.norm(estimator_classical2.theta_fit))

        plt.plot(xtest, mu,  lw=3, label='full '+str(rank), color=color)

        plt.plot(xtest, mu_classical, lw=3, label='classical', color='yellow', linestyle = '-')
        #plt.plot(xtest, mu_classical, lw=3, label='classical2 ' + str(rank), color='orange', linestyle = '--')

        # plt.fill_between(xtest.view(-1), mu.view(-1) - std_det.view(-1), mu.view(-1) + std_det.view(-1), color=color,
        #                  alpha=.1)
        #
        # plt.fill_between(xtest.view(-1), mu.view(-1) - std_det2.view(-1), mu.view(-1) + std_det2.view(-1), color='tab:red',
        #                  alpha=.1)

        lcb = estimator.lcb(xtest)
        ucb = estimator.ucb(xtest)

        # print (mu)
        # print (lcb)
        plt.fill_between(xtest.view(-1), lcb.view(-1), ucb.view(-1), color=color, alpha=.1)

        plt.plot(Xtrain, ytrain, 'ko', lw=3)
        plt.plot(xtest, f(xtest), 'k--', lw=3)

    plt.legend()
    plt.show()
