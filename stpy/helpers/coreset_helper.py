import torch
import numpy as np
from stpy.borel_set import BorelSet
from stpy.kernels import KernelFunction
from stpy.continuous_processes.gauss_procc import GaussianProcess
from doexpy.general_search.Exploration import GreedyExploration
def epsilon_net(borel_set, k):
	pass

def coreset(borel_set, k):

	pass

def coreset_leverage_score_greedy(borel_set,kernel,n, tol = 10e-4):
	xtest = borel_set.return_discretization(n)
	k = kernel.kernel
	N = xtest.size()[0]
	score = 1
	K = k(xtest,xtest)
	x = xtest[torch.randint(0,N,(1,)),:].view(1,-1)
	c = 1
	while score> tol:
		I = torch.eye(c).double()
		scores = np.diag(K - k(xtest,x).T@torch.pinverse(k(x,x)+tol*I)@k(x,xtest).T)
		index = np.argmax(scores)
		x = torch.cat((x,xtest[index,:].view(1,-1)))
		score = scores[index]
		c = c + 1
	return x

def coreset_greedy(borel_set,kernel,n, s):
	xtest = borel_set.return_discretization(n)
	N = xtest.size()[0]
	GP = GaussianProcess(kernel = k)
	x = xtest[torch.randint(0,N,(1,)),:].view(1,-1)
	Greedy = GreedyExploration(x,lambda x: torch.sum(x*0,dim = 1),GP, 0.)
	for i in range(s):
		Greedy.step(xtest)
	return GP.x


if __name__ == "__main__":

	B = BorelSet(2,torch.Tensor([[-1,1],[-1,1]]).double())
	#B = BorelSet(1, torch.Tensor([[-1, 1]]).double())
	k = KernelFunction(gamma = 1., d = 2)
	n = 10
	coreset = coreset_leverage_score_greedy(B,k,n, tol = 10e-2)
	#coreset = coreset_greedy(B,k,n,10)

	import matplotlib.pyplot as plt
	xtest = B.return_discretization(n)
	plt.scatter(xtest[:, 0], xtest[:, 1], color = 'red')
	plt.scatter(coreset[:,0],coreset[:,1], marker='s')
	plt.show()

