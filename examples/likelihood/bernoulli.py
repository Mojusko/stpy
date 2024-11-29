from itertools import repeat
#import matplotlib.pyplot as plt
import warnings
import argparse
import numpy as np
import torch
from scipy import optimize

from stpy.regression.regularized_dictionary.regularized_dictionary import RegularizedDictionary
from stpy.embeddings.nystrom_fea import NystromFeatures
from stpy.embeddings.embedding import HermiteEmbedding, Embedding
from stpy.test_functions.benchmarks import Simple1DFunction, SwissFEL, StybTangBenchmark, CamelbackBenchmark
from stpy.probability.bernoulli_likelihood import BernoulliLikelihoodCanonical
from stpy.probability.noise_models import BernoulliNoise,PoissonNoise

from stpy.regularization.regularizer import L2Regularizer
from stpy.kernel import KernelFunction
from stpy.probability.poisson_likelihood import PoissonLikelihoodCanonical
from stpy.helpers.helper import interval_torch
from stpy.test_functions.benchmarks import Simple1DFunction

if __name__ == "__main__":
    n = 1024
    N = 20
    x = interval_torch(N, d = 1)
    # space of actions
    xtest = interval_torch(n, d= 1)  # pytorch array (n, d)

    benchmark_f = Simple1DFunction(d=1)
    # scale
    kappa = torch.max(torch.abs(benchmark_f.eval_noiseless(xtest)))

    # reard of the best action

    print("Variation:", kappa)
    mu = lambda x: torch.sigmoid(x)
    noise_model = BernoulliNoise(prob=lambda x: mu(benchmark_f.eval_noiseless(x)))


    # lambda interface to the black-box function
    def F(x):
        return noise_model.sample_noise(x)


    def F_noise_less(x):
        return noise_model.mean(x)


    best = torch.max(F_noise_less(xtest))
    print("The highest payoff:", best)

    # define features for the linear kernel
    bound = 10.
    lam = 1./bound

    embedding = HermiteEmbedding(m=128, gamma=0.1, d=1, kappa=kappa)
    likelihood = BernoulliLikelihoodCanonical()
    regularizer = L2Regularizer(lam=lam)
    estimator = RegularizedDictionary(embedding, likelihood, regularizer)


    y = F(x)
    estimator.load_data((x,y))
    estimator.fit()

    ytest = estimator.mean(xtest)
    y_ground_truth = F_noise_less(xtest)

    #plot
    import matplotlib.pyplot as plt
    plt.plot(xtest.numpy(), F_noise_less(xtest).numpy(), 'b--', label = 'True function (sigmoid)')
    plt.plot(xtest.numpy(), mu(ytest).numpy(), 'r--',label =  'Estimated function (sigmoid)')
    plt.plot(x.numpy(),y.numpy(),'ko', label = 'Data')
    #plt.plot(xtest.numpy(), ytest.numpy(), 'r--', 'Estimated parametrized')

    plt.legend()
    plt.show()
