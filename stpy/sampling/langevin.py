import numpy as np
import torch
import scipy

class LangevinSampler():

	def __init__(self, verbose = False):
		self.verbose = verbose
		pass

	def calculate(self, HessianF,theta0):
		W = HessianF(theta0)
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-3))
		return L

	def sample(self, F, nablaF, HessianF, theta0, steps = 100):
		L = self.calculate(HessianF, theta0)
		eta = 0.5 / (L + 1)
		m = theta0.size()[0]
		theta = theta0
		for k in range(steps):
			w = torch.randn(size=(m, 1)).double()
			theta = theta - eta * nablaF(theta) + np.sqrt(2 * eta) * w
			if self.verbose == True:
				print("Iter:", k, theta.T)
		return theta