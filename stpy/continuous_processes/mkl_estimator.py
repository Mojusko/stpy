import cvxpy as cp
import numpy as np
import torch

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.regularization.regularizer import Regularizer
from stpy.regularization.simplex_regularizer import DirichletRegularizer, SupRegularizer

class MultipleKernelLearner(GaussianProcess):

	def __init__(self, kernel_objects,
				 lam: float =1.0,
				 s:  float = 0.01,
				 opt: str = 'closed',
				 regularizer: Regularizer = None):

		self.kernel_objects = kernel_objects
		self.no_models = len(kernel_objects)
		self.regularizer = regularizer
		self.s = s
		self.lam = lam
		self.opt = opt

		self.var = 'fixed'

	def fit(self):
		self.fit_gp(self.x,self.y)

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
					A = self.Ks[i] * alpha[i]
				else:
					A += self.Ks[i] * alpha[i]
			A = A + np.eye(self.n)*self.lam*self.s**2
			constraints = []
			l = cp.reshape(u, (1, 1))
			G = cp.bmat([[A, y.numpy()], [y.numpy().T, l]])
			constraints += [G >> 0]
			constraints += [alpha >= 0.]
			constraints += [cp.sum(alpha) == 1.]

			objective = cp.Minimize( u)
			prob = cp.Problem(objective, constraints)
			prob.solve( solver = cp.MOSEK,verbose = True)

		elif self.opt == "closed":
			alpha = cp.Variable(self.no_models, nonneg=True)
			A = sum([self.Ks[i] * alpha[i] for i in range(self.no_models)])+ np.eye(self.n) * self.lam * self.s ** 2
			constraints = [cp.sum(alpha)==1, alpha<=1]
			objective = cp.matrix_frac(self.y.numpy(), A)
			if self.regularizer is not None and self.regularizer.is_convex():
				objective = objective + self.regularizer.get_regularizer_cvxpy()(alpha)
				prob = cp.Problem(cp.Minimize(objective), constraints)
				prob.solve(solver=cp.MOSEK, verbose=False)

			elif self.regularizer is not None and not self.regularizer.is_convex():
				obj,con,vars = self.regularizer.get_cvxpy_objectives_constraints_variables(self.no_models)
				no_problems = len(con)
				vals = []
				args = []
				for i in range(no_problems):
					prob = cp.Problem(cp.Minimize(objective+obj[i](alpha,*vars)), constraints + con[i](alpha, *vars))
					prob.solve(solver=cp.MOSEK, verbose=False)
					vals.append(prob.value)
					args.append(alpha.value)
				alpha.value = args[np.argmin(vals)]
			else:
				prob = cp.Problem(cp.Minimize(objective), constraints)
				prob.solve(solver=cp.MOSEK, verbose=False)

		self.alphas = torch.from_numpy(alpha.value)
		if self.regularizer is not None:
			print (self.regularizer.name, self.alphas)
		else:
			print("No", self.alphas)
		self.K = torch.sum(torch.stack([alpha*K for alpha,K in zip(self.alphas, self.Ks)]), dim = 0) + np.eye(self.n)*self.lam*self.s**2
		self.fitted = True

	def execute(self, xtest):
		if self.fitted == True:
			Ks = [self.kernel_objects[i].kernel(self.x, xtest) for i in range(self.no_models)]
			K_star = torch.sum(torch.stack([alpha * K for alpha, K in zip(self.alphas, Ks)]), dim=0)
		else:
			K_star = None
		Ks = [self.kernel_objects[i].kernel(xtest, xtest) for i in range(self.no_models)]
		K_star_star = torch.sum(torch.stack([alpha * K for alpha, K in zip(self.alphas, Ks)]), dim=0)
		return (K_star, K_star_star)

	# def log_marginal(self, kernel, X, weight):
	# 	pass

	def mean(self, xtest):
		K_star, K_star_star = self.execute(xtest)
		self.A = torch.linalg.lstsq(self.K, self.y)[0]
		ymean = torch.mm(K_star, self.A)
		return ymean

	def mean_std(self, xtest, full=False, reuse=False):
		K_star, K_star_star = self.execute(xtest)
		self.A = torch.linalg.lstsq(self.K, self.y)[0]
		ymean = torch.mm(K_star, self.A)

		if self.var == 'fixed':
			ystd = self.std_fixed(xtest)
		elif self.var == 'true':
			ystd = self.std_opt(xtest)
		return (ymean, ystd)

	def lcb(self, xtest: torch.Tensor, type=None, arg=False, sign=1.):
		theta = cp.Variable((self.alpha, 1))
		args = []
		n = xtest.size()[0]
		values = torch.zeros(size=(n, 1)).double()
		Phi = self.embed(xtest)

		for j in range(n):
			objective = sign * Phi[j, :] @ theta
			if (self.constraints is not None and not self.constraints.is_convex()):
				value, theta_lcb = self.objective_on_non_convex_confidence_set(theta, objective, type=type)
			elif not self.regularizer.is_convex():
				value, theta_lcb = self.objective_on_non_convex_confidence_set_bisection(theta, objective,
																						 type=type)
			else:
				value, theta_lcb = self.objective_on_confidence_set(theta, objective, type=type)

			values[j] = sign * value
			if arg:
				args.append(theta_lcb)

		if args:
			return values, args
		else:
			return values

	def ucb(self, xtest):
		pass

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

	def sample(self, xtest, size=1):
		pass

if __name__ == "__main__":
	from stpy.continuous_processes.gauss_procc import GaussianProcess
	from stpy.helpers.helper import interval_torch
	import matplotlib.pyplot as plt
	n = 512
	N = 5
	s = 0.1
	d = 1

	xtest = interval_torch(n,d)
	x = interval_torch(N,d)

	kernel1 = KernelFunction(gamma = 0.05)
	kernel2 = KernelFunction(kernel_name="polynomial", power = 5)
	kernel3 = KernelFunction(kernel_name="polynomial", power=3)
	kernel4 = KernelFunction(kernel_name="polynomial", power=2)
	kernel5 = KernelFunction(kernel_name="polynomial", power=1)
	kernel6 = KernelFunction(kernel_name="polynomial", power=1)

	kernels = [kernel1, kernel2,kernel3, kernel4, kernel5, kernel6]

	GP = GaussianProcess(kernel=kernel1)
	torch.manual_seed(2)
	y = GP.sample(x)

	# sup inverse barrier
	for lam in [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.9999]:
		regularizer = SupRegularizer(d = len(kernels), lam = lam, constrained=True, version='1')
		mkl = MultipleKernelLearner(kernels, regularizer= regularizer)
		mkl.fit_gp(x,y)
		mkl.visualize(xtest, size = 0, show = False, fig = False, color = 'tab:blue', label = " sup:"+str(lam))
		regularizer = SupRegularizer(d=len(kernels), lam=lam, constrained=True, version='2')
		mkl = MultipleKernelLearner(kernels, regularizer=regularizer)
		mkl.fit_gp(x, y)
		mkl.visualize(xtest, size=0, show=False, fig=False, color='tab:green', label=" sup:" + str(lam))

	# dirichlet mixture
	regularizer = DirichletRegularizer(d=len(kernels), lam=lam, constrained=True)
	mkl = MultipleKernelLearner(kernels, regularizer=regularizer)
	mkl.fit_gp(x, y)
	mkl.visualize(xtest, size=0, show=False, fig=False, color='tab:red', label = " dirichlet")

	# no regularizer
	mkl = MultipleKernelLearner(kernels, regularizer=None)
	mkl.fit_gp(x, y)
	mkl.visualize(xtest, size=0, show=False, fig=False, color='tab:orange', label = " no")

	plt.show()
