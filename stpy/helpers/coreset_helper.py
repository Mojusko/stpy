import numpy as np
import torch


def epsilon_net(borel_set, k):
	pass


def coreset(borel_set, k):
	pass


def coreset_leverage_score_greedy(borel_set, kernel, n, tol=10e-4):
	xtest = borel_set.return_discretization(n)
	k = kernel.kernel
	N = xtest.size()[0]
	score = 1
	K = k(xtest, xtest)
	x = xtest[torch.randint(0, N, (1,)), :].view(1, -1)
	c = 1
	while score > tol:
		I = torch.eye(c).double()
		scores = np.diag(K - k(xtest, x).T @ torch.pinverse(k(x, x) + tol * I) @ k(x, xtest).T)
		index = np.argmax(scores)
		x = torch.cat((x, xtest[index, :].view(1, -1)))
		score = scores[index]
		c = c + 1
	return x
