import copy
from itertools import repeat
#import matplotlib.pyplot as plt
import warnings
import argparse
import numpy as np
import torch
from scipy import optimize

from stpy.regression.regularized_dictionary.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.polynomial_embedding import PolynomialEmbedding, ChebyschevEmbedding
from stpy.probability.noise_models import BernoulliNoise
from stpy.probability.gaussian_likelihood import GaussianLikelihood
from stpy.regularization.regularizer import L2Regularizer
from stpy.regularization.group_regularizer import GroupRegularizer

from stpy.helpers.helper import interval_torch
from stpy.test_functions.benchmarks import Simple1DFunction, CustomBenchmark


if __name__ == "__main__":

    repeats = 10
    n = 1024

    N = 1
    T = 21
    x = interval_torch(N, d = 1)

    h1 = []
    h2 = []
    h3 = []
    h4 = []

    for repeat in range(repeats):
        # space of actions
        xtest = interval_torch(n, d= 1)  # pytorch array (n, d)

        benchmark_f = CustomBenchmark(d=1, func = lambda x: x**1 + 1.)
        # scale
        kappa = 1 #torch.max(torch.abs(benchmark_f.eval_noiseless(xtest)))

        print ("Kappa", kappa)
        sigma = 0.5

        def F_noise_less(x):
            return benchmark_f.eval_noiseless(x)

        def F(x):
            return F_noise_less(x) + torch.randn(x.size()[0]).view(-1,1)*sigma

        best = torch.max(F_noise_less(xtest))
        print("The highest payoff:", best)


        embedding = ChebyschevEmbedding(kappa = kappa, d = 1, p = 10)
        m = embedding.get_m()

        # define features for the linear kernel
        bound = 1.
        lam = 1. / bound

        likelihood = GaussianLikelihood(sigma = sigma)
        base_regularizer = L2Regularizer(lam=lam)

        base_constraint = base_regularizer.get_constraint_object(1.)

        #groups = [torch.arange(0, i).tolist() for i in range(1,m)]
        groups = [[i] for i in range(0,m)]

        # groups = []
        # for i in range(1,m):
        #     mask = torch.zeros(m, dtype=torch.bool)
        #     mask[torch.arange(0, i).tolist()] = True
        #     groups.append(mask)

        regularizer = GroupRegularizer(groups, m, base_regularizer = base_regularizer, active=2)
        regularizer_classic = L2Regularizer(lam=lam)

        verbose = False
        tol = 1e-8

        # sparse
        estimator  = RegularizedDictionary(embedding, likelihood, regularizer, constraints=base_constraint, inference_type='LR', check  = "none", use_constraint=True, tolerance=tol, verbose=verbose)

        # non-sparse
        estimator2 = RegularizedDictionary(embedding, copy.deepcopy(likelihood), regularizer_classic, constraints=base_constraint, inference_type = 'LR', check = "none", use_constraint=True, tolerance=tol, verbose=verbose)

        # non-sparse
        estimator3 = RegularizedDictionary(embedding, copy.deepcopy(likelihood), regularizer_classic, constraints=base_constraint, inference_type='posterior-prior-LR', check="none", use_constraint=True, tolerance=tol, verbose=verbose)

        # sparse
        estimator4 = RegularizedDictionary(embedding, copy.deepcopy(likelihood), regularizer, constraints=base_constraint, inference_type='posterior-prior-LR', check="none",  use_constraint=True, tolerance=tol, verbose=verbose)

        y = F(x)

        estimator.load_data((x,y))
        estimator2.load_data((x,y))
        estimator3.load_data((x,y))
        estimator4.load_data((x,y))

        # iterate and add 2N datapoints one by one

        for i in range(T):

            # sample random point
            xnext = torch.rand(1).view(1,1)*2 - 1

            # evaluate it
            ynext = F(xnext)

            estimator.add_points((xnext,ynext))
            estimator2.add_points((xnext, ynext))
            estimator3.add_points((xnext, ynext))
            estimator4.add_points((xnext, ynext))

            estimator.fit()
            estimator2.fit()
            estimator3.fit()
            estimator4.fit()

            ytest = estimator.mean(xtest)
            ytest2 = estimator2.mean(xtest)
            ytest3 = estimator3.mean(xtest)
            ytest4 = estimator4.mean(xtest)

            y_ground_truth = F_noise_less(xtest)

            I = torch.eye(m).double()
            Es = torch.vstack([I[group,:].view(1,-1) for group in groups])

            lcbs1 = estimator.lcb(Es, embeded=True)
            val1  = estimator.map_raw(Es)
            ucbs1 = estimator.ucb(Es, embeded=True)

            lcbs2 = estimator2.lcb(Es, embeded=True)
            val2 = estimator2.map_raw(Es)
            ucbs2 = estimator2.ucb(Es, embeded=True)

            lcbs3 = estimator3.lcb(Es, embeded=True)
            val3 = estimator3.map_raw(Es)
            ucbs3 = estimator3.ucb(Es, embeded=True)

            lcbs4 = estimator4.lcb(Es, embeded=True)
            val4 = estimator4.map_raw(Es)
            ucbs4 = estimator4.ucb(Es, embeded=True)

            #plot
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(xtest.numpy(), F_noise_less(xtest).numpy(), 'b-', label = 'True function')
            plt.plot(xtest.numpy(), ytest.numpy(), '--',color = 'purple',label =  'Estimated function - Sparse 1 ')
            plt.plot(xtest.numpy(), ytest2.numpy(), '--',color = 'red',label =  'Estimated function - Non-sparse 2')
            plt.plot(xtest.numpy(), ytest3.numpy(), '--',color = 'blue', label='Estimated function - Non-sparse 3 ')
            plt.plot(xtest.numpy(), ytest4.numpy(), '--', color = 'green', label='Estimated function - sparse 4')
            plt.plot(estimator.x.numpy(),estimator.y.numpy(),'ko', label = 'Data')
            plt.legend()
            #plt.show()

            bar_width = 0.2  # Width of bars
            offset = [-1, 0, 1, 2]  # Offsets for grouping bars
            xx = np.arange(len(groups))

            height1 = ucbs1 - lcbs1
            height2 = ucbs2 - lcbs2
            height3 = ucbs3 - lcbs3
            height4 = ucbs4 - lcbs4

            if i == 20:
                h1.append(height1)
                h2.append(height2)
                h3.append(height3)
                h4.append(height4)

            if repeat == 1:

                plt.figure()
                #
                # plt.bar(xx.reshape(-1) + offset[0] * bar_width, height2.reshape(-1), bar_width, bottom=lcbs2.reshape(-1),  color='red', alpha=1, label='EMK - non-sparse prior')  # Use 'bottom' to start from y1
                # #plt.plot(xx.reshape(-1) + offset[0] * bar_width, val2.reshape(-1), color='red', linestyle='', marker = 'o')
                #
                # plt.bar(xx.reshape(-1) + offset[1] * bar_width, height1.reshape(-1), bar_width, bottom=lcbs1.reshape(-1),  color='red', alpha=0.3, label='EMK - sparse prior')  # Use 'bottom' to start from y1
                # #plt.plot(xx.reshape(-1) + offset[1] * bar_width, val1.reshape(-1), color= 'purple', linestyle='', marker = 'o')  # Use 'bottom' to start from y1

                plt.bar(xx.reshape(-1) + offset[2] * bar_width, height3.reshape(-1), bar_width, bottom=lcbs3.reshape(-1),  color='blue', alpha=1, label='MQ - non-sparse prior') # Use 'bottom' to start from y1
                #plt.plot(xx.reshape(-1) + offset[2] * bar_width, val3.reshape(-1), color='blue', linestyle='', marker = 'o')

                plt.bar(xx.reshape(-1) + offset[3] * bar_width, height4.reshape(-1), bar_width, bottom=lcbs4.reshape(-1),  color='blue', alpha=0.3, label='MQ - sparse prior')  # Use 'bottom' to start from y1
                #plt.plot(xx.reshape(-1) + offset[3] * bar_width, val4.reshape(-1), color='green', linestyle='', marker = 'o')
                weights = np.zeros(m)
                weights[0] = 1.
                weights[1] = 1.
                plt.plot(xx + offset[0]*bar_width ,weights, 'ko', label = 'True')
                plt.legend()
                plt.savefig(f"pics-poly/figure{i}.png", dpi = 100)

    print (torch.stack(h1).size())
    h1 = [torch.nan_to_num(h, nan=1e2) for h in h1]
    h2 = [torch.nan_to_num(h, nan=1e2) for h in h2]

    plt.figure()

    # plt.bar(xx.reshape(-1) + offset[0] * bar_width, torch.sum(torch.stack(h2)[:,:,0], dim = 0), bar_width, bottom=0,
    #         color='red', alpha=1, label='EMK - non-sparse prior')  # Use 'bottom' to start from y1
    # # plt.plot(xx.reshape(-1) + offset[1] * bar_width, val2.reshape(-1), color='red', linestyle='', marker='o')
    # plt.errorbar(xx.reshape(-1) + offset[0] * bar_width,
    #              torch.sum(torch.stack(h2)[:, :, 0], dim = 0), yerr=torch.std(torch.stack(h2)[:,:,0], dim=0),
    #                        fmt='none', ecolor='black', capsize=5, capthick=1, elinewidth=1)
    #
    #
    # plt.bar(xx.reshape(-1) + offset[1] * bar_width, torch.sum(torch.stack(h1)[:,:,0], dim = 0), bar_width, bottom=0,
    #         color='red', alpha=0.5, label='EMK - sparse prior')  # Use 'bottom' to start from y1
    # # plt.plot(xx.reshape(-1) + offset[0] * bar_width, val1.reshape(-1), color='purple', linestyle='',
    # #          marker='o')  # Use 'bottom' to start from y1
    # plt.errorbar(xx.reshape(-1) + offset[1] * bar_width, torch.sum(torch.stack(h1)[:,:,0],dim = 0),
    #              yerr=torch.std(torch.stack(h1)[:,:,0], dim=0),
    #              fmt='none', ecolor='black', capsize=5, capthick=1, elinewidth=1)

    plt.bar(xx.reshape(-1) + offset[2] * bar_width, torch.sum(torch.stack(h3)[:,:,0], dim = 0), bar_width, bottom=0,
            color='blue', alpha=1, label='MQ - non-sparse prior')  # Use 'bottom' to start from y1
    # plt.plot(xx.reshape(-1) + offset[2] * bar_width, val3.reshape(-1),   color='blue', linestyle='', marker='o')
    plt.errorbar(xx.reshape(-1) + offset[2] * bar_width,
                 torch.sum(torch.stack(h3)[:, :, 0], dim = 0), yerr=torch.std(torch.stack(h3)[:,:,0], dim=0),
                           fmt='none', ecolor='black', capsize=5, capthick=1, elinewidth=1)

    plt.bar(xx.reshape(-1) + offset[3] * bar_width, torch.sum(torch.stack(h4)[:,:,0], dim = 0), bar_width, bottom=0,
            color='blue', alpha=0.5, label='MQ - sparse prior')  # Use 'bottom' to start from y1
    plt.errorbar(xx.reshape(-1) + offset[3] * bar_width,
                 torch.sum(torch.stack(h4)[:, :, 0], dim = 0), yerr=torch.std(torch.stack(h4)[:,:,0], dim=0),
                           fmt='none', ecolor='black', capsize=5, capthick=1, elinewidth=1)

    # plt.plot(xx.reshape(-1) + offset[3] * bar_width, val4.reshape(-1), color='green', linestyle='', marker='o')
    weights = np.zeros(m)
    weights[0] = 1.
    weights[1] = 1.
    plt.ylabel("Confidence set size")
    plt.xlabel("Parameter index")
    # plt.plot(xx + offset[0] * bar_width, weights, 'ko', label='True Parameters')
    plt.legend()
    plt.savefig(f"avg_bars.png")