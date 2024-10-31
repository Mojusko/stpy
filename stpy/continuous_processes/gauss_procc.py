import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
import torch
from cvxpylayers.torch import CvxpyLayer
#from functorch import hessian
import functorch
from pymanopt.manifolds import Euclidean, Stiefel, PSDFixedRank
from torch.autograd import grad
from torchmin import minimize as minimize_torch

import stpy.helpers.helper as helper
from stpy.estimator import Estimator
from stpy.kernels import KernelFunction


class GaussianProcess(Estimator):

	def __init__(self, gamma=1, s=0.001, kappa=1., kernel_name="squared_exponential", diameter=1.0,
				 groups=None, bounds=None, nu=1.5, kernel=None, d=1, power=2, lam=1., loss = 'squared', huber_delta = 1.35,
				 hyper = 'classical', B = 1., svr_eps = 0.1):
		"""

		:param gamma: Smoothnes parameter for squared exponential, laplace and matern kernel
		:param s: level of noise
		:param kernel: choose from a list
		:param diameter: diameter of the set (deprecated)
		:param groups: additive groups
		:param bounds: bounds for the continuous optimization
		:param v: parameter for matern kernel
		"""

		## GP properties
		self.s = s
		self.d = d
		self.x = None
		self.K = np.array([1.0])
		self.mu = 0.0
		self.lam = lam
		self.total_bound = B
		self.prob = 0.5
		self.svr_eps = svr_eps
		self.safe = False
		self.fitted = False
		self.diameter = diameter
		self.bounds = bounds
		self.admits_first_order = False
		self.back_prop = True
		self.loss = loss
		self.huber_delta = huber_delta
		self.hyper = hyper
		self.prepared_log_marginal = False
		self.warm_start_solution = None
		self.max_size = 10000
		## kernel hyperparameters
		if kernel is not None:
			self.kernel_object = kernel
			self.kernel = kernel.kernel
			self.d = kernel.d
		else:
			self.kernel_object = KernelFunction(kernel_name=kernel_name, gamma=gamma, nu=nu, groups=groups, kappa=kappa,
												power=power, d=d)
			self.kernel = self.kernel_object.kernel

			self.gamma = gamma
			self.v = nu
			self.groups = groups
			self.kappa = kappa
			self.custom = kernel
			self.optkernel = kernel_name

	def residuals(self,x,y):
		res = (self.mean(x) - y)
		return res

	def description(self):
		"""
		Description of GP in text
		:return: string with description
		"""
		return self.kernel_object.description() + "\nlambda=" + str(self.s)

	def embed(self, x):
		return self.kernel_object.embed(x)

	def get_basis_size(self):
		return self.kernel_object.get_basis_size()

	def make_safe(self, x):
		"""
		Make the input dataset numerically stable by removing duplicates?
		:param x:
		:return:
		"""
		self.epsilon = 0.001
		# remove vectors that are very close to each other
		return x

	def add_data_point(self, x, y, Sigma = None):

		if self.x is not None:
			self.x = torch.cat((self.x, x), dim=0)
			self.y = torch.cat((self.y, y), dim=0)
			if Sigma is None:
				self.Sigma = torch.block_diag(self.Sigma, torch.eye(x.size()[0],dtype = torch.double) * self.s)
			else:
				self.Sigma = torch.block_diag(self.Sigma, Sigma)
		else:
			self.x = x
			self.y = y
			self.Sigma = Sigma
		self.fit_gp(self.x, self.y, Sigma = self.Sigma)

	def fit(self, x=None, y=None):
		if x is not None:
			self.fit_gp(x,y)
		else:
			self.fit_gp(self.x,self.y, Sigma=self.Sigma)

	def lcb(self, xtest):
		"""
		Lower confidence bound
		:return:
		"""
		mu, s = self.mean_std(xtest)
		return mu - 2 * s

	def ucb(self, xtest):
		"""
		Upper confidence bound
		:param xtest:
		:return:
		"""
		mu, s = self.mean_std(xtest)
		return mu + 2*s

	def fit_gp(self, x, y, Sigma = None, iterative=False, extrapoint=False):
		"""
		Fits the Gaussian process, possible update is via iterative inverse
		:param x: data x
		:param y: values y
		:param iterative: iterative inverse, where only last point of x is used
		:param extrapoint: iterative inverse must be allowed, x is the only addition
		:return:
		"""
		# first fit
		try:
			self.n, self.d = list(x.size())
		except:
			self.n, self.d = x.shape

		if Sigma is None:
			self.Sigma = (self.s) * torch.eye(self.n, dtype=torch.float64)
		else:
			self.Sigma = Sigma

		if (self.fitted == False or iterative == False):

			if self.safe == True:
				x = self.make_safe(x)

			self.x = x
			self.y = y
			self.K = self.kernel(x, x) + self.Sigma.T @ self.Sigma
			self.fitted = True
		else:
			# iterative inverse
			if (iterative == True):
				if extrapoint == False:
					last_point = self.x[-1, :].view(1, -1)
				else:
					last_point = x
				old_K = self.K
				old_Kinv = self.Kinv
			else:
				pass
		self.mean_std(x)
		return None

	def norm(self):
		if self.fitted:
			val = torch.sqrt(self.A.T @ self.kernel(self.x, self.x) @ self.A)
			return val
		else:
			return None

	def beta(self, delta=1e-3, norm=1):
		"""
		return concentration parameter given the current estimates

		:param delta: failure probability
		:param norm: norm assumption
		:return:
		"""
		beta_value = self.s * norm + \
					 torch.sqrt(2 * torch.log(1. / delta + torch.log(torch.det(self.K) / self.s ** self.n)))
		return beta_value

	def execute(self, xtest):
		"""
		Calculates the covariance between data and xtest
		:param xtest:
		:return:
		"""
		if self.fitted == True:
			K_star = self.kernel(self.x, xtest)
		else:
			K_star = None
		K_star_star = self.kernel(xtest, xtest)
		return (K_star, K_star_star)

	def _huber_fit(self, K_star, newK = None):
		alpha = cp.Variable(self.n)
		self.jitter = 10e-5
		if newK is None:
			K = self.kernel(self.x, self.x) + self.jitter * torch.eye(self.n, dtype=torch.float64)
		else:
			K = newK.detach()
		K = cp.atoms.affine.wraps.psd_wrap(K)
		objective = cp.Minimize(cp.sum(cp.huber((K @ alpha - self.y.view(-1).numpy())/self.s,M = self.huber_delta)) + self.lam * cp.quad_form(alpha, K))
		prob = cp.Problem(objective)
		prob.solve(solver = cp.MOSEK, enforce_dpp = False)
		if K_star is not None:
			return K_star@torch.from_numpy(alpha.value).view(-1,1)
		else:
			return torch.from_numpy(alpha.value).view(-1,1)

	def _svr_fit(self, K_star, newK = None):
		alpha = cp.Variable(self.n)
		self.jitter = 10e-5
		if newK is None:
			K = self.kernel(self.x, self.x) + self.jitter * torch.eye(self.n, dtype=torch.float64)
		else:
			K = newK.detach()

		K = cp.atoms.affine.wraps.psd_wrap(K)
		objective = cp.Minimize(self.lam * cp.quad_form(alpha, K))
		constraints = [cp.abs(K @ alpha - self.y.view(-1).numpy()) <= self.svr_eps ]
		prob = cp.Problem(objective, constraints)
		prob.solve(solver = cp.MOSEK, enforce_dpp = False)
		if K_star is not None:
			return K_star@torch.from_numpy(alpha.value).view(-1,1)
		else:
			return torch.from_numpy(alpha.value).view(-1,1)


	def _unif_fit(self, K_star, newK = None):
		alpha = cp.Variable((self.n,1))
		self.jitter = 10e-5
		if newK is None:
			K = self.kernel(self.x, self.x) + self.jitter * torch.eye(self.n, dtype=torch.float64)
		else:
			K = newK.detach()

		K = cp.atoms.affine.wraps.psd_wrap(K)
		con = 2*self.total_bound*self.prob/((1-self.prob)*np.sqrt(2*np.pi*self.s**2))
		objective = cp.Minimize(cp.sum(cp.logistic(cp.square(
			(K @ alpha - self.y.view(-1, 1).numpy())/ (np.sqrt(2)*self.s)) + np.log(con) )) + self.lam * cp.quad_form(alpha, K))
		prob = cp.Problem(objective)
		prob.solve(solver = cp.MOSEK, enforce_dpp = False)
		if K_star is not None:
			return K_star@torch.from_numpy(alpha.value).view(-1,1)
		else:
			return torch.from_numpy(alpha.value).view(-1,1)

	def _unif_fit_torch(self, K_star, newK = None, warm_start = None):
		self.jitter = 10e-5
		if newK is None:
			K = self.kernel(self.x, self.x) + self.jitter * torch.eye(self.n, dtype=torch.float64)
		else:
			K = newK.detach()

		con = 2 * self.total_bound * self.prob / ((1 - self.prob) * np.sqrt(2 * np.pi * self.s ** 2))
		unif = lambda alpha: torch.sum(torch.log(torch.exp( ((K@alpha-self.y.view(-1))**2)/(2*self.s**2) + np.log(con) ) + 1 ) ) \
										  + self.lam * alpha  @ K@ alpha
		if warm_start is None:
			x_init = torch.zeros(size = (self.n,1)).view(-1).double()
		else:
			x_init = warm_start.view(-1)

		res = minimize_torch(unif, x_init, method='l-bfgs', tol=1e-3, disp=0,
							 options={'max_iter': 200, 'gtol': 1e-3})
		alpha = res.x

		if K_star is not None:
			return K_star @ alpha.view(-1, 1)
		else:
			return alpha.view(-1, 1)

	def _huber_fit_torch(self, K_star, newK = None):
		self.jitter = 10e-5
		if newK is None:
			K = self.kernel(self.x, self.x) + self.jitter * torch.eye(self.n, dtype=torch.float64)
		else:
			K = newK
		L = torch.linalg.cholesky(K)

		huber = lambda beta: torch.nn.functional.huber_loss(L @ beta / self.s, self.y.view(-1) / self.s,
															reduction='sum',
															delta=self.huber_delta) + self.lam * beta @ beta
		#x_init = torch.linalg.solve(L.T@L+torch.eye(self.n).double()*self.s**2*self.lam, self.y)
		x_init = torch.zeros(size = (self.n,1)).view(-1).double()
		res = minimize_torch(huber, x_init, method='l-bfgs', tol=1e-4, disp=0,
							 options={'max_iter': 10**3, 'gtol': 1e-4})
		alpha = torch.linalg.solve(L,res.x)
		if K_star is not None:
			return K_star @ alpha.view(-1, 1)
		else:
			return alpha.view(-1,1)

	def mean_std(self, xtest, full=False, reuse=False):
		if xtest.size()[0]<self.max_size:
			return self.mean_std_sub(xtest,full=full, reuse=reuse)
		else:
			stepby = self.max_size
			mu = torch.zeros(size=(xtest.size()[0], 1)).double()
			std = torch.zeros(size=(xtest.size()[0], 1)).double()

			# first
			i = 0
			mu[i * stepby:(i + 1) * stepby], std[i * stepby:(i + 1) * stepby] = self.mean_std_sub(
				xtest[i * stepby:(i + 1) * stepby, :], reuse=False)

			for i in np.arange(1, xtest.size()[0] // stepby, 1):
				print(i, "/", xtest.size()[0] // stepby)
				mu[i * stepby:(i + 1) * stepby], std[i * stepby:(i + 1) * stepby] = self.mean_std_sub(
					xtest[i * stepby:(i + 1) * stepby, :], reuse=True)

			# last
			if xtest.size()[0] % stepby > 0:
				mu[xtest.size()[0] - xtest.size()[0] % stepby:], std[
															  xtest.size()[0] - xtest.size()[0] % stepby:] = self.mean_std_sub(
					xtest[xtest.size()[0] - xtest.size()[0] % stepby:, :], reuse=True)

			return mu, std

	def mean_std_sub(self, xtest, full=False, reuse=False):
		"""
		Return posterior mean and variance as tuple
		:param xtest: grid, numpy array (2D)
		:param full: Instead of just poinwise variance, full covariance can be outputed (bool)
		:return: (tensor,tensor)
		"""
		if full:
			(K_star, K_star_star) = self.execute(xtest)
		else:
			K_star = self.kernel(self.x, xtest)
			diag_K_star_star = torch.hstack([self.kernel(xtest[i,:].view(1,-1),xtest[i,:].view(1,-1)).view(1) for i in range(xtest.size()[0])])

		if self.fitted == False:
			# the process is not fitted

			if full == False:
				x = torch.sum(xtest, dim=1)
				#first = torch.diag(K_star_star).view(-1, 1)
				first = diag_K_star_star.view(-1,1)
				variance = first
				yvar = torch.sqrt(variance)
			else:
				x = torch.sum(xtest, dim=1)
				first = K_star_star
				yvar = first

			return (0 * x.view(-1, 1), yvar)

		else:

			if self.back_prop == False:
				if reuse == False:
					#self.decomp = torch.lu(self.K.unsqueeze(0))
					self.LU, self.pivot = torch.linalg.lu_factor(self.K.unsqueeze(0))
					#self.A = torch.lu_solve(self.y.unsqueeze(0), *self.decomp)[0, :, :]
					self.A = torch.linalg.lu_solve(self.LU, self.pivot, self.y.unsqueeze(0))[0,:,:]
				self.B = torch.t(torch.linalg.lu_solve(self.LU, self.pivot ,torch.t(K_star).unsqueeze(0))[0, :, :])
			else:
				if reuse == False:
					self.A = torch.linalg.lstsq(self.K, self.y)[0]
				#self.B = torch.t(torch.linalg.solve(self.K, torch.t(K_star)))
				self.B = torch.t(torch.linalg.lstsq(self.K, torch.t(K_star))[0])

			if self.loss == "squared":
				ymean = torch.mm(K_star, self.A)
			elif self.loss == "huber":
				ymean = self._huber_fit(K_star)
			elif self.loss == "svr":
				ymean = self._svr_fit(K_star)
			elif self.loss == "unif"  or self.loss == "unif_new":
				ymean = self._unif_fit_torch(K_star)
			else:
				raise AssertionError("Loss function not implemented.")

			if full == False:
				first = diag_K_star_star.view(-1,1)
				second = torch.einsum('ij,ji->i', (self.B, torch.t(K_star))).view(-1, 1)
				variance = first - second
				yvar = torch.sqrt(variance)
			else:
				first = K_star_star
				second = torch.mm(self.B, torch.t(K_star))
				yvar = first - second

			return (ymean, yvar)

	def mean(self, xtest):
		"""
		Calculates the mean prediction over a specific input space
		:param xtest: input
		:return:
		"""
		K_star = self.kernel(self.x, xtest)

		if self.loss == "squared":
			ymean = torch.mm(K_star, self.A)
		elif self.loss == "huber":
			ymean = self._huber_fit(K_star)
		else:
			raise AssertionError("Loss function not implemented.")

		return ymean

	def gradient_mean_var(self, point, hessian=True):
		"""
		Can calculate gradient at single point atm.

		:param point:
		:return:
		"""

		# mean
		point.requires_grad_(True)
		mu = self.mean_std(point)[0]
		nabla_mu = grad(mu, point, create_graph=True)[0][0]

		if hessian == True:
			# variance
			H = self.kernel_object.get_2_der(point)
			C = self.kernel_object.get_1_der(point, self.x)

			V = H - torch.t(C) @ self.K @ C

			return [nabla_mu, V]
		else:
			return nabla_mu

	def mean_gradient_hessian(self, xtest, hessian=False):
		hessian_mu = torch.zeros(size=(self.d, self.d), dtype=torch.float64)
		xtest.requires_grad_(True)
		# xtest.retain_grad()
		mu = self.mean_std(xtest)[0]
		# mu.backward(retain_graph=True)

		# nabla_mu = xtest.grad
		nabla_mu = grad(mu, xtest, create_graph=True)[0][0]

		if hessian == False:
			return nabla_mu
		else:
			for i in range(self.d):
				hessian_mu[i, :] = grad(nabla_mu[i], xtest, create_graph=True, retain_graph=True)[0][0]
			return [nabla_mu, hessian_mu]

	def sample(self, xtest, size=1, jitter=10e-8):
		"""
			Samples Path from GP, return a numpy array evaluated over grid
		:param xtest: grid
		:param size: number of samples
		:return: numpy array
		"""
		nn = list(xtest.size())[0]

		if self.fitted == True:
			(ymean, yvar) = self.mean_std(xtest, full=True)
			Cov = yvar + 10e-10 * torch.eye(nn, dtype=torch.float64)
			L = torch.linalg.cholesky(Cov)
			# L = torch.from_numpy(np.linalg.cholesky(Cov.numpy()))
			random_vector = torch.normal(mean=torch.zeros(nn, size, dtype=torch.float64), std=1.)
			f = ymean + torch.mm(L, random_vector)
		else:
			(K_star, K_star_star) = self.execute(xtest)
			L = torch.linalg.cholesky(K_star_star + jitter * torch.eye(nn, dtype=torch.float64))
			random_vector = torch.normal(mean=torch.zeros(nn, size, dtype=torch.float64), std=1.)
			f = self.mu + torch.mm(L, random_vector)
		return f

	def sample_and_max(self, xtest, size=1):
		"""
			Samples Path from GP and takes argmax
		:param xtest: grid
		:param size: number of samples
		:return: (argmax, max)
		"""
		f = self.sample(xtest, size=size)
		self.temp = f
		val, index = torch.max(f, dim=0)
		return (xtest[index, :], val)


	def log_marginal(self, kernel, X, weight):

		if self.loss == "squared":
			return self._log_marginal_squared(kernel, X, weight)
		elif self.loss == "unif_new":
			return self._log_marginal_unif(kernel, X, weight)
		else:
			return self._log_marginal_map(kernel, X, weight)

	def _log_marginal_unif(self,kernel,X,weight):
		if not self.prepared_log_marginal:
			self._prepare_log_marginal_unif()

		func = kernel.get_kernel()
		self.jitter = 10e-4
		K = func(self.x, self.x, **X) + torch.eye(self.n, dtype=torch.float64) * self.jitter
		#print ("Kernel")
		#print (K)
		L = torch.linalg.cholesky(K)
		self.L_unif.value = (L.data.numpy())

		self.prob_unif.solve(solver=cp.MOSEK, enforce_dpp=False, warm_start=True)

		solution = torch.zeros(size=(self.n, 1), requires_grad=True).reshape(-1).double()
		solution.data = torch.from_numpy(self.beta_unif.value)
		con = 2 * self.total_bound * self.prob / ((1 - self.prob) * np.sqrt(2 * np.pi * self.s ** 2))

		loglikelihood = lambda beta: torch.sum(torch.log(torch.exp( ((L@beta-self.y.view(-1))**2)/(2*self.s**2) + np.log(con) ) + 1 ) ) \
										  + self.lam * beta.T  @ beta

		H = hessian(loglikelihood)(solution)
		logdet = - 0.5* torch.slogdet(H)[1] * weight
		logprob = -0.5* loglikelihood(solution) + logdet
		logprob = -logprob
		return logprob

	def _prepare_log_marginal_unif(self):

		self.beta_unif = cp.Variable(self.n)
		self.L_unif = cp.Parameter((self.n, self.n))

		con = 2 * self.total_bound * self.prob / ((1 - self.prob) * np.sqrt(2 * np.pi * self.s ** 2))
		#self.objective_unif = cp.Minimize(cp.sum(cp.logistic(cp.square(
		#	(self.K_unif @ self.alpha_unif - self.y.view(-1).numpy()) / (np.sqrt(2) * self.s)) + np.log(con))) + self.lam * cp.quad_form(
		#	self.alpha_unif, self.L))
		self.objective_unif = cp.Minimize(cp.sum(cp.logistic(cp.square(
			(self.L_unif @ self.beta_unif - self.y.view(-1).numpy()) / (np.sqrt(2) * self.s)) + np.log(con))) + self.lam * cp.sum_squares(self.beta_unif))
		self.prob_unif = cp.Problem(self.objective_unif)
		self.prepared_log_marginal  = True

	def _prepare_log_marginal_huber(self):
		beta = cp.Variable(self.n)
		L = cp.Parameter((self.n, self.n))

		objective = cp.Minimize(cp.sum(
			cp.huber((L @ beta - self.y.view(-1).numpy()) / self.s, M=self.huber_delta)) + self.lam * cp.sum_squares(
			beta))

		prob = cp.Problem(objective)
		cvxpylayer = CvxpyLayer(prob, parameters=[L], variables=[beta])
		self.prepared_log_marginal = True
		print ("cvxpy-layer has been initialized.")
		return cvxpylayer

	def _log_marginal_huber_cvxpy(self, kernel, X, weight):
		func = kernel.get_kernel()
		self.jitter = 10e-4
		L_tch = torch.linalg.cholesky(func(self.x, self.x, **X) + torch.eye(self.n, dtype=torch.float64) * self.jitter)

		if not self.prepared_log_marginal:
			self._cvxpylayer = self._prepare_log_marginal_huber()
		solution = self._cvxpylayer(L_tch)[0]

		huber = lambda beta: torch.nn.functional.huber_loss(L_tch@beta/self.s,self.y.view(-1)/self.s,reduction='sum',delta = self.huber_delta) + self.lam * beta.T @ beta
		H = torch.autograd.functional.hessian(huber, solution)

		logdet = - 0.5* torch.slogdet(H)[1]* weight
		logprob = -0.5* huber(solution) +logdet
		logprob = -logprob
		return logprob


	def _log_marginal_map(self, kernel, X, weight):
		# this implementation uses Danskin theorem to simplify gradient propagation
		func = kernel.get_kernel()
		self.jitter = 10e-4
		K_tch =func(self.x, self.x, **X) + torch.eye(self.n, dtype=torch.float64) * self.jitter

		# solve
		solution = torch.zeros(size=(self.n, 1), requires_grad=True).reshape(-1).double()
		if self.warm_start_solution is None:
			self.warm_start_solution = solution.clone()

		if self.loss == "huber":
			alpha = self._huber_fit(None, newK = K_tch).detach()
			loglikelihood = lambda alpha: torch.nn.functional.huber_loss(K_tch@alpha/self.s,self.y.view(-1)/self.s,
									reduction='sum',delta = self.huber_delta) + self.lam * alpha.T @K_tch@ alpha

			solution.data = alpha.reshape(-1).data
			self.warm_start_solution.data = solution.data
			mask = torch.abs(K_tch @ alpha - self.y)/self.s<self.huber_delta
			mask = mask.view(-1).double()
			D = torch.diag(mask)
			H =  K_tch@D@K_tch+ 2 * self.lam * K_tch

		elif self.loss == "svr":
			alpha = self._svr_fit(None, newK=K_tch).detach()

			loglikelihood = lambda alpha: torch.sum(torch.abs(K_tch@alpha-self.y.view(-1))*(K_tch@alpha -self.y.view(-1) > self.svr_eps).int()) \
										 + self.lam * alpha.T @K_tch@ alpha

			solution.data = alpha.reshape(-1).data
			self.warm_start_solution.data = solution.data
			H = torch.autograd.functional.hessian(loglikelihood, solution)

		elif self.loss == "unif":
			alpha = self._unif_fit_torch(None, newK=K_tch).detach()
			con = 2 * self.total_bound * self.prob / ((1 - self.prob) * np.sqrt(2 * np.pi * self.s ** 2))


			loglikelihood = lambda alpha: torch.sum(torch.log(torch.exp( ((K_tch@alpha-self.y.view(-1))**2)/(2*self.s**2) + np.log(con) ) + 1 ) ) \
										  + self.lam * alpha @ K_tch@ alpha
			#v = lambda alpha : torch.sum(torch.exp( ((K_tch@alpha-self.y.view(-1))**2)/(2*self.s**2) + np.log(con) ))
			solution.data = alpha.reshape(-1).data
			self.warm_start_solution.data = solution.data
			H = hessian(loglikelihood)(solution)

		logdet = - 0.5* torch.slogdet(H)[1] * weight
		logprob = -0.5* loglikelihood(solution) + logdet
		logprob = -logprob
		return logprob



	def _log_marginal_squared(self, kernel, X, weight):
		func = kernel.get_kernel()
		K = func(self.x, self.x, **X) + torch.eye(self.n, dtype=torch.float64) * self.s * self.s
		logdet = -0.5 * torch.slogdet(K)[1] * weight
		alpha = torch.linalg.solve(K, self.y)
		logprob = -0.5 * torch.mm(torch.t(self.y), alpha) + logdet
		logprob = -logprob
		return logprob

	def optimize_params(self, type='bandwidth', restarts=10, regularizer=None,
						maxiter=1000, mingradnorm=1e-4, verbose=False, optimizer="pymanopt", scale=1., weight=1., save = False,
								save_name = 'model.np', init_func = None, bounds = None, parallel = False, cores = None):

		# Spectral norm regularizer
		if regularizer is not None:
			if regularizer[0] == "spectral_norm":
				regularizer_func = lambda S: regularizer[1] * torch.norm(1/S[0], p='nuc')
			elif regularizer[0] == 'lasso':
				regularizer_func = lambda S: regularizer[1] * torch.norm(1/S[0], p=1)
			else:
				regularizer_func = None
		else:
			regularizer_func = None

		if type == "bandwidth":
			params = {}
			for key, dict2 in self.kernel_object.params_dict.items():
				if 'gamma' in dict2.keys():
					params[key] = {'gamma': (init_func, Euclidean(1), bounds)}
				elif 'ard_gamma' in dict2.keys():
					params[key] = {'ard_gamma': (init_func, Euclidean(len(dict2['group'])), bounds)}

		elif type == "bandwidth+noise":
			params = {}
			init_func_noise = lambda x: self.s
			for key, dict2 in self.kernel_object.params_dict.items():

				if 'gamma' in dict2.keys():
					params[key] = {'gamma': (init_func, Euclidean(1), bounds)}

				elif 'ard_gamma' in dict2.keys():
					params[key] = {'ard_gamma': (init_func, Euclidean(len(dict2['group'])), bounds)}

			params['likelihood'] = {'sigma':(init_func_noise, Euclidean(1), None )}

		elif type == "rots":
			params = {}
			d = int(self.kernel_object.d)
			for key, dict2 in self.kernel_object.params_dict.items():
				if 'rot' in dict2.keys():
					params[key] = {'rot': (None, Stiefel(d, d), None)}
		elif type == "groups":
			params = {}
			optimizer = "discrete"
			d = self.kernel_object.d
			for key, dict2 in self.kernel_object.params_dict.items():
				if 'groups' in dict2.keys():
					params[key] = {'groups': (None, helper.generate_groups(d), None)}
			pass
		elif type == "covariance":
			params = {}
			d = int(self.kernel_object.d)
			for key, dict2 in self.kernel_object.params_dict.items():
				if 'cov' in dict2.keys():
					params[key] = {'cov': (None, PSDFixedRank(d, d), None)}
		else:
			raise AttributeError("This quick-optimization is not implemented.")

		self.optimize_params_general(params=params, restarts=restarts,
									 optimizer=optimizer, regularizer_func=regularizer_func,
									 maxiter=maxiter, mingradnorm=mingradnorm, verbose=verbose, scale=scale,
									 weight=weight, save = save, save_name = save_name, parallel = parallel, cores = cores)

	def log_probability(self, xtest, sample):
		from scipy.stats import multivariate_normal
		mu, covar = self.mean_std(xtest, full=True)
		p = np.log(multivariate_normal.pdf(sample.view(-1).numpy(), mean=mu.view(-1).numpy(), cov=covar.numpy()))
		return p

	def volume_mean_cvxpy(self, xtest, weights=None, eps=10e-2,
						  tol=10e-14, max_weight=1, max_iter=1000,
						  verbose=False, scale=10e-4, slope=1.,
						  bisections=10, B='auto', optimal_scale=None,
						  optimize_scale=False, relax='relu'):

		n = self.x.size()[0]
		K = self.get_kernel()  # (self.x, self.x)
		Kinv = torch.pinverse(K + eps * torch.eye(K.size()[0]).double()).numpy()
		if weights is None:
			weights = torch.ones(self.x.size()[0]) / n
		if B == 'auto':
			alpha, _ = torch.lstsq(self.y, K)
			beta = K @ alpha
			B = beta.T @ Kinv @ beta
			print("Auto:B", B)

		def fun(scale_arg):
			beta = cp.Variable(n)
			if relax == 'relu':
				loss_fn_transformed = cp.sum(cp.pos(weights * slope * (
							cp.abs(beta - self.y.numpy().reshape(-1)) - eps))) + 0.5 * scale_arg * cp.quad_form(beta,
																												Kinv)
			elif relax == 'log':
				loss_fn_transformed = cp.sum(cp.logistic(weights * slope * (
						cp.abs(beta - self.y.numpy().reshape(-1)) - eps))) + 0.5 * scale_arg * cp.quad_form(beta,
																											Kinv)

			# loss_fn_transformed = cp.sum(weights*logit(slope*(cp.abs(beta - self.y.numpy().reshape(-1)) -eps))) +  0.5*scale_arg*cp.quad_form(beta, Kinv)-

			prob = cp.Problem(cp.Minimize(loss_fn_transformed))
			# prob.solve(solver=cp.MOSEK, feastol=tol, verbose=False)
			prob.solve(solver=cp.MOSEK, verbose=False)
			if verbose == True:
				print("scale:", scale_arg, "cond:", np.linalg.cond(Kinv), "sub.", beta.value.T @ Kinv @ beta.value - B,
					  "B:", B)
			return beta.value.T @ Kinv @ beta.value - B

		if optimize_scale:
			return helper.bisection(fun, 0., max_weight, bisections)

		if optimal_scale is None:
			scale_star = helper.bisection(fun, 0., max_weight, bisections)
		else:
			scale_star = optimal_scale

		beta = cp.Variable(n)
		if relax == 'relu':
			loss_fn_transformed = cp.sum(weights * cp.pos(
				slope * (cp.abs(beta - self.y.numpy().reshape(-1)) - eps))) + 0.5 * scale_star * cp.quad_form(beta,
																											  Kinv)
		elif relax == 'log':
			loss_fn_transformed = cp.sum(weights * cp.logistic(
				slope * (cp.abs(beta - self.y.numpy().reshape(-1)) - eps))) + 0.5 * scale_star * cp.quad_form(beta,
																											  Kinv)
		prob = cp.Problem(cp.Minimize(loss_fn_transformed))
		# prob.solve(solver=cp.CVXOPT, feastol=tol, verbose=verbose)
		prob.solve(solver=cp.MOSEK, verbose=verbose)
		beta_torch = torch.from_numpy(beta.value).view(-1, 1)
		alpha = torch.from_numpy(Kinv) @ beta_torch
		ytest = self.kernel(self.x, xtest) @ alpha
		return ytest

	def volume_mean(self, xtest, weights=None, eps=10e-2, tol=10e-6, max_iter=1000, verbose=False, eta_start=0.01,
					eta_decrease=0.9, scale=1, slope=1., warm=True, relax='relu', norm=False, B='auto'):
		self.scale = scale
		self.relax = relax

		K = self.get_kernel()  # (self.x, self.x)
		Kinv = torch.pinverse(K)

		if weights is None:
			weights = torch.ones(self.x.size()[0])
		else:
			weights[weights < 10e-6] = 0.  # * self.x.size()[0]
			weights = weights.view(-1)
		if warm == True:
			# warm start with L2 fit
			alpha, _ = torch.lstsq(self.y, K)
			beta = K @ alpha
		else:
			beta = torch.randn(size=(self.n, 1)).double()  # .requires_grad_(True)*0

		# loss_fn_original = lambda alpha: torch.sum(torch.relu(torch.abs(K @ alpha - self.y) -eps)) + 0.5*self.s * alpha.T @ K @ alpha
		if self.relax == "relu":
			loss_fn_transformed = lambda beta: torch.sum(
				torch.relu(torch.abs(beta - self.y) - eps)) + self.scale * 0.5 * self.s * beta.T @ Kinv @ beta

		elif self.relax == "tanh":
			self.slope = slope
			tanh = lambda x: (torch.tanh(self.slope * x) + 1) * 0.5
			loss_fn_transformed = lambda beta: torch.sum(weights * tanh(torch.abs(beta - self.y) - eps).view(
				-1)) + 0.5 * self.s * self.scale * beta.T @ Kinv @ beta

		elif self.relax == "elu":
			self.slope = slope
			elu = lambda x: torch.nn.elu(x, alpha=self.slope)
			loss_fn_transformed = lambda beta: torch.sum(
				elu(torch.abs(beta - self.y) - eps)) + 0.5 * self.s * self.scale * beta.T @ Kinv @ beta

		elif self.relax == "relu":
			return self.volume_mean_cvxpy(xtest, weights=weights, eps=eps, scale=scale, tol=tol)
		else:
			raise AssertionError("Unkown relaxation.")

		current_loss = 10e10
		eta = eta_start
		for i in range(max_iter):
			grad = self.s * (Kinv @ beta)
			beta = self.proximal(beta, grad, eta, eps, weights)
			past_loss = current_loss
			current_loss = loss_fn_transformed(beta)
			if current_loss > past_loss:
				eta = eta * eta_decrease
			elif np.abs(current_loss - past_loss) < tol:
				break

			# print (i, beta.T)
			if verbose == True:
				print(i, loss_fn_transformed(beta), eta)

		print("final norm:", beta.T @ Kinv @ beta)

		# alpha = torch.inverse(self.K) @ beta
		alpha = torch.pinverse(K) @ beta
		# alpha = torch.lstsq(K,beta)
		ytest = self.kernel(self.x, xtest) @ alpha
		# max = torch.max(torch.abs(beta - self.y))
		if norm == True:
			return beta.T @ Kinv @ beta
		# yz = self.kernel(self.x, self.x)  @ alpha
		# approx_v = torch.sum(torch.relu(torch.abs(beta - self.y) -eps))/max
		# approx_p = approx_v/self.n
		# mask = (torch.abs(yz[:,0] - self.y[:,0])) > eps
		# approx_p = float(torch.sum(mask))/float(self.n)
		return ytest  # ,approx_p

	def volume_mean_norm(self, xtest, weights=None, eps=10e-2, tol=10e-6, max_iter=1000, verbose=False, eta_start=0.01,
						 eta_decrease=0.9, scale=1, slope=1., warm=True, relax='relu', B='auto'):
		K = self.kernel(self.x, self.x)
		Kinv = torch.pinverse(K)
		if B == 'auto':
			alpha, _ = torch.lstsq(self.y, self.K)
			beta = K @ alpha
			B = beta.T @ Kinv @ beta

		func = lambda s: self.volume_mean(xtest, weights=weights, eps=eps, tol=tol, max_iter=max_iter, verbose=verbose,
										  eta_start=eta_start,
										  eta_decrease=eta_decrease, scale=s, slope=slope, warm=warm, relax=relax,
										  norm=True) - B

		s_star = stpy.optim.custom_optimizers.bisection(func, 0., 1000., 10)

		return self.volume_mean(xtest, weights=weights, eps=eps, tol=tol, max_iter=max_iter, verbose=verbose,
								eta_start=eta_start,
								eta_decrease=eta_decrease, scale=s_star, slope=slope, warm=warm, relax=relax,
								norm=False)

	def proximal(self, beta, nabla, eta, eps, weights):
		res = beta
		for i in range(self.n):
			from scipy.optimize import minimize

			b = float(beta[i, :])
			y = float(self.y[i, :])
			g = float(nabla[i, :])
			w = float(weights[i])
			# s = float(self.s)

			tanh = lambda x: (np.tanh(self.slope * x) + 1) * 0.5
			elu = lambda x: torch.elu(x, alpha=self.slope).numpy()

			if self.relax == "relu":
				loss_reg = lambda x: w * np.maximum(0, np.abs(x - y) - eps)
			elif self.relax == "tanh":
				loss_reg = lambda x: w * tanh(np.abs(x - y) - eps)
			elif self.relax == "elu":
				loss_reg = lambda x: w * elu(np.abs(x - y) - eps)
			else:
				raise AssertionError("Unkown relaxation.")

			loss_scalar = lambda x: ((1 / (2. * eta)) * (x - (b - eta * g)) ** 2) + loss_reg(x)

			x0 = np.array([0.])
			# print (minimize(loss_scalar,x0,method ='nelder-mead').x)
			res[i, :] = float(minimize(loss_scalar, x0, method='nelder-mead').x)
		return res

	def get_lambdas(self, beta, mean=False):
		"""
		Gets lambda function to evaluate acquisiton function and its derivative
		:param beta: beta in GP-UCB
		:return: [lambda,lambda]
		"""
		mean = lambda x: self.mean_std(x.reshape(1, -1), reuse=True)[0][0][0]
		sigma = lambda x: self.mean_std(x.reshape(1, -1), reuse=True)[1][0][0]

		if mean == True:
			return [mean, sigma]
		else:
			fun = lambda x: -(mean(x) + np.sqrt(beta) * sigma(x))
			grad = lambda x: -complex_step_derivative(fun, 1e-10, x.reshape(1, -1))

			return [fun, grad]

	def get_kernel(self):
		return self.K

	def ucb_optimize(self, beta, multistart=25, lcb=False):
		"""
		Optimizes UCB acquisiton function and return next point and its value as output
		:param beta: beta from GP UCB
		:param multistart: number of starts
		:return: (next_point, value at next_point)
		"""

		mean = lambda x: self.mean_std(x, reuse=True)[0][0][0]
		sigma = lambda x: self.mean_std(x, reuse=True)[1][0][0]

		ucb = lambda x: torch.dot(torch.Tensor([1.0, np.sqrt(beta)]), torch.Tensor(
			[self.mean_std(x, reuse=True)[0][0][0], self.mean_std(x, reuse=True)[1][0][0]]))
		lcb = lambda x: torch.dot(torch.Tensor([1.0, np.sqrt(beta)]), torch.Tensor(
			[self.mean_std(x, reuse=True)[0][0][0], -self.mean_std(x, reuse=True)[1][0][0]]))

		if lcb == False:
			fun2 = lambda x: -ucb(torch.from_numpy(x).view(1, -1)).numpy()
		else:
			fun2 = lambda x: -lcb(torch.from_numpy(x).view(1, -1)).numpy()
		fun = lambda x: -(
					mean(torch.from_numpy(x).view(1, -1)) + np.sqrt(beta) * sigma(torch.from_numpy(x).view(1, -1)))

		self.back_prop = False
		self.mean_std(self.x)

		mybounds = self.bounds

		results = []

		from scipy.optimize import minimize

		for i in range(multistart):
			x0 = np.random.randn(self.d)
			for i in range(self.d):
				x0[i] = np.random.uniform(mybounds[i][0], mybounds[i][1])

			res = minimize(fun2, x0, method="L-BFGS-B", jac=None, tol=0.000001, bounds=mybounds)
			solution = res.x
			results.append([solution, -fun(solution)])

		results = np.array(results)
		index = np.argmax(results[:, 1])
		solution = results[index, 0]

		return (torch.from_numpy(solution), -fun(solution))

	def isin(self, xnext):
		self.epsilon = 0.001
		for v in self.x:
			if torch.norm(v - xnext, p=2) < self.epsilon:
				return True

	def sample_and_condition(self, x):
		xprobe = x.view(1, -1)
		fprobe = self.sample(xprobe)
		if not self.isin(xprobe):
			self.x = torch.cat((self.x, xprobe), dim=0)
			self.y = torch.cat((self.y, fprobe), dim=0)
			self.fit_gp(self.x, self.y)
		return -fprobe

	def get_lambdas_TH(self):
		fun = lambda x: self.sample_and_condition(x)
		grad = None
		return [fun, grad]

	def sample_iteratively_max(self, xtest, multistart=20, minimizer="coordinate-wise", grid=100):
		"""
			Samples Path from GP and takes the maximum iteratively
			:param xtest: grid
			:param size: number of samples
			:return: numpy array
		"""
		# print ("Iterative:",multistart,minimizer,grid)
		from scipy.optimize import minimize
		# old stuff
		xold = self.x
		yold = self.y

		# with fixed grid
		if xtest is not None:
			# number of samples
			nn = xtest.shape[0]

			f = torch.zeros(nn, dtype=torch.float64)

			for j in range(nn):
				xprobe = xtest[j, :].view(1, -1)
				(K_star, K_star_star) = self.execute(xprobe)
				(ymean, yvar) = self.mean_std(xprobe)
				L = torch.sqrt(K_star_star + self.s * self.s * torch.eye(1, dtype=torch.float64) - yvar)
				fprobe = ymean + L * torch.randn(1, dtype=torch.float64)
				# add x and fprobe to the dataset and redo the whole
				f[j] = fprobe
				if not self.isin(xprobe):
					self.x = torch.cat((self.x, xprobe), dim=0)
					self.y = torch.cat((self.y, fprobe), dim=0)

				self.fit_gp(self.x, self.y)

			val, index = torch.max(f, dim=0)
			self.fit_gp(xold, yold)
			return (xtest[index, :], f[index])

		else:
			# Iterative without grid

			# get bounds
			if self.bounds == None:
				mybounds = tuple([(-self.diameter, self.diameter) for i in range(self.d)])
			else:
				mybounds = self.bounds
			[fun, grad] = self.get_lambdas_TH()

			results = []
			for j in range(multistart):

				# print ("Multistart:",j)
				x0 = torch.randn(self.d, dtype=torch.float64)
				for i in range(self.d):
					x0[i].uniform_(mybounds[i][0], mybounds[i][1])

				# simple coordnate-wise optimization
				if minimizer == "coordinate-wise":
					solution = x0
					for i in range(self.d):
						xtest = torch.from_numpy(np.tile(x0, (grid, 1)))
						xtest[:, i] = torch.linspace(mybounds[i][0], mybounds[i][1], grid)
						sample = self.sample(xtest)

						## Add to the posterior
						self.x = torch.cat((self.x, xtest), dim=0)
						self.y = torch.cat((self.y, sample), dim=0)

						# argmax
						val, index = torch.max(sample, dim=0)
						out = xtest[index, :]

						# fit new GP
						self.fit_gp(self.x, self.y)
						solution[i] = out[0, i]

				elif minimizer == "L-BFGS-B":
					solution = np.random.randn(self.d)
					xmax = [b[1] for b in mybounds]
					xmin = [b[0] for b in mybounds]
					bounds = MyBounds(xmax=xmax, xmin=xmin)
					func = lambda x: fun(torch.from_numpy(x)).numpy()[0][0]
					res = scipy.optimize.basinhopping(func, solution, disp=False, niter=grid, accept_test=bounds)
					solution = torch.from_numpy(res.x)

				else:
					raise AssertionError("Wrong optimizer selected.")

				results.append(torch.cat((solution, -fun(solution)[0])))
				self.x = xold
				self.y = yold
				self.fit_gp(self.x, self.y)

			results = torch.stack(results)
			val, index = torch.max(results[:, -1], dim=0)
			solution = results[index, 0:self.d].view(1, self.d)
			self.x = xold
			self.y = yold
			self.fit_gp(self.x, self.y)

			return (solution, -fun(solution))


if __name__ == "__main__":
	from stpy.helpers.helper import interval
	# domain size
	L_infinity_ball = 1
	# dimension
	d = 1
	# error variance
	s = torch.from_numpy(np.array(1.0, dtype=np.float64))

	# grid density
	n = 1024
	# number of intial points
	N = 32
	# smoothness
	gamma = 0.1
	# test problem

	xtest = torch.from_numpy(interval(n, d))
	# x = torch.from_numpy(np.random.uniform(-L_infinity_ball,L_infinity_ball, size = (N,d)))
	x = torch.from_numpy(interval(N, 1))
	f_no_noise = lambda q: torch.sin(torch.sum(q * 4, dim=1)).view(-1, 1)
	f = lambda q: f_no_noise(q) + torch.normal(mean=torch.zeros(q.size()[0], 1, dtype=torch.float64), std=1.,
											   out=None) * s * s
	# targets
	y = f(x)

	# GP model with squared exponential
	kernel = KernelFunction(kernel_name = "ard", gamma = torch.ones(d, dtype = torch.float64)*gamma , groups = [[0],[1]])
	# kernel = KernelFunction(kernel_name="ard", gamma=torch.ones(1, dtype=torch.float64) * gamma, groups=[[0]])
	GP = GaussianProcess(s=s, d=1)

	# fit GP
	# x = x.numpy()
	GP.fit_gp(x, y)
	# get mean and variance of GP
	[mu, std] = GP.mean_std(xtest)

	# print ("Log probability:", GP.log_marginal_likelihood() )
	# mu_inf = GP.chebyshev_mean(xtest)
	eps = 0.1

	mu_vol = GP.volume_mean_cvxpy(xtest, eps=eps, verbose=True, scale=1., slope=1., tol=10e-9)

	GP.visualize(xtest, f_true=f_no_noise, show=False)
	plt.plot(xtest.numpy(), mu_vol.detach().numpy(), label="Least-Volume-ReLu", lw=2)
	for slope in [0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000.]:
		# mu_vol_log = GP.volume_mean_cvxpy(xtest, eps=eps, verbose=True, scale=1., slope=slope, tol=10e-9, relax = 'log', B = 1000)
		# plt.plot(xtest.numpy(),mu_vol_log.detach().numpy(), '--',label = "Least-Volume-Log" + str(slope), lw = 2)
		mu_vol_tanh = GP.volume_mean(xtest, eps=eps, verbose=True, eta_start=0.1, eta_decrease=0.1, scale=1.,
									 slope=slope,
									 tol=0.01, warm=True, relax='tanh')
		plt.plot(xtest.numpy(), mu_vol_tanh.detach().numpy(), '-.', label="Least-Volume-Tanh" + str(slope), lw=2)
	# print (slope, np.sum(np.abs(mu_vol_log)<eps) )
	# plt.plot(xtest.numpy(),mu_vol_tanh.detach().numpy(), label = "Least-Volume-Tahn", lw = 2)
	# plt.plot(xtest.numpy(),mu_vol_tanh2.detach().numpy(), label = "Least-Volume-Tahn2", lw = 2)

	# plt.plot(xtest.numpy(),mu_inf.detach().numpy(), label = "Chebyschev estimate", lw = 2)
	plt.plot(x.numpy(), y.numpy() + eps, 'ko')
	plt.plot(x.numpy(), y.numpy() - eps, 'ko')
	plt.legend()
	plt.show()
