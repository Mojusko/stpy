import numpy as np
import cvxpy as cp
import numpy as np
import torch

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction


class MultipleKernelLearner(GaussianProcess):

	def __init__(self, kernel_objects, lam=1.0, s=0.01, opt = 'sdp'):
		self.kernel_objects = kernel_objects
		self.no_models = len(kernels)
		self.s = s
		self.lam = lam
		self.opt = opt
		self.var = 'fixed'

	def fit_gp(self, x, y):
		self.x = x
		self.y = y
		(self.n, self.d) = self.x.size()

		self.Ks = []
		for i in range(self.no_models):
			self.Ks.append(self.kernel_objects[i].kernel(x,x))

		if self.opt == 'sdp':
			alpha = cp.Variable(self.no_models)
			u = cp.Variable(1)
			A = None
			for i in range(self.no_models):
				if A is None:
					A = self.Ks[i] * alpha[i] + np.eye(self.n) * self.lam * self.s ** 2
				else:
					A += self.Ks[i]*alpha[i] + np.eye(self.n)*self.lam*self.s**2
			constraints = []
			l = cp.reshape(u, (1, 1))
			G = cp.bmat([[A, y.numpy()], [y.numpy().T, l]])
			constraints += [G >> 0]
			constraints += [alpha >= 0.]
			constraints += [cp.sum(alpha) == 1.]

			objective = cp.Minimize( u)
			prob = cp.Problem(objective, constraints)
			prob.solve( solver = cp.MOSEK,verbose = True)

		self.alphas = torch.from_numpy(alpha.value)
		self.K = torch.sum(torch.stack([alpha*K for alpha,K in zip(self.alphas, self.Ks)]), dim = 0) + np.eye(self.n)*self.lam*self.s**2
		self.fit = True
		print (self.alphas)

	def execute(self, xtest):
		if self.fit == True:
			Ks = [self.kernel_objects[i].kernel(self.x, xtest) for i in range(self.no_models)]
			K_star = torch.sum(torch.stack([alpha * K for alpha, K in zip(self.alphas, Ks)]), dim=0)
		else:
			K_star = None
		Ks = [self.kernel_objects[i].kernel(xtest, xtest) for i in range(self.no_models)]
		K_star_star = torch.sum(torch.stack([alpha * K for alpha, K in zip(self.alphas, Ks)]), dim=0)
		return (K_star, K_star_star)

	def log_marginal(self, kernel, X, weight):
		pass

	def mean_std(self, xtest, full=False, reuse=False):
		K_star, K_star_star = self.execute(xtest)
		self.A = torch.linalg.lstsq(self.K, self.y)[0]
		ymean = torch.mm(K_star, self.A)

		if self.var == 'fixed':
			ystd = self.std_fixed(xtest)
		elif self.var == 'true':
			ystd = self.std_opt(xtest)
		return (ymean, ystd)

	def std_opt(self, xtest):
		N = xtest.size()[0]
		for i in range(N):
			x = xtest[i,:]
			theta = cp.Variable(self.n*self.no_models)
			M = torch.block_diag(self.Ks)
			cp.norm(theta,p=2)*theta[i]




	def std_fixed(self, xtest):
		K_star, K_star_star = self.execute(xtest)
		self.B = torch.t(torch.linalg.solve(self.K, torch.t(K_star)))
		first = torch.diag(K_star_star).view(-1, 1)
		second = torch.einsum('ij,ji->i', (self.B, torch.t(K_star))).view(-1, 1)
		variance = first - second
		ystd = torch.sqrt(variance)
		return ystd

	def mean(self, xtest):
		return self.mean_std(xtest)[0]

	def sample(self, xtest, size=1):
		pass

if __name__ == "__main__":
	from stpy.continuous_processes.gauss_procc import GaussianProcess
	from stpy.helpers.helper import interval_torch
	n = 512
	N = 12
	s = 0.01
	d = 1

	xtest = interval_torch(n,d)
	x = interval_torch(N,d)
	kernel1 = KernelFunction(gamma = 0.1)
	kernel2 = KernelFunction(kernel_name="polynomial")
	kernels = [kernel1, kernel2]

	GP = GaussianProcess(kernel=kernel2)
	y = GP.sample(x)

	mkl = MultipleKernelLearner(kernels)
	mkl.fit_gp(x,y)

	mkl.visualize(xtest, size = 0)
