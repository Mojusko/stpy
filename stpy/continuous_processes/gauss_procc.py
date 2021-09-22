import numpy as np
import scipy as scipy
import torch
import cvxpy as cp
import matplotlib.pyplot as plt
from stpy.kernels import KernelFunction
from stpy.random_process import RandomProcess
import stpy.helpers.helper as helper
from torch.autograd import grad
import pymanopt
from pymanopt.manifolds import Euclidean, Product, Stiefel, PSDFixedRank
from pymanopt.solvers import SteepestDescent
from scipy.optimize import minimize

class GaussianProcess(RandomProcess):

	def __init__(self,gamma = 1 ,s=0.001, kappa = 1., kernel_name = "squared_exponential", diameter = 1.0,
				 groups = None, bounds = None, nu = 2, kernel = None, d  = 1 , power = 2, lam = 1.):
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
		self.safe = False
		self.fit = False
		self.diameter = diameter
		self.bounds = bounds
		self.admits_first_order = False
		self.back_prop = True

		## kernel hyperparameters
		if kernel is not None:
			self.kernel_object = kernel
			self.kernel = kernel.kernel
			self.d = kernel.d
		else:
			self.kernel_object = KernelFunction(kernel_name = kernel_name, gamma = gamma, nu = nu, groups = groups, kappa = kappa, power = power, d = d)
			self.kernel = self.kernel_object.kernel

			self.gamma = gamma
			self.v = nu
			self.groups = groups
			self.kappa = kappa
			self.custom = kernel
			self.optkernel = kernel_name




	def description(self):
		"""
		Description of GP in text
		:return: string with description
		"""
		return self.kernel_object.description() + "\nlambda=" + str(self.s)

	def embed(self,x):
		return self.kernel_object.embed(x)

	def get_basis_size(self):
		return self.kernel_object.get_basis_size()

	def make_safe(self,x):
		"""
		Make the input dataset numerically stable by removing duplicates?
		:param x:
		:return:
		"""
		self.epsilon = 0.001
		# remove vectors that are very close to each other
		return x

	def add_data_point(self,x,y):

		if self.x is not None:
			self.x = torch.cat((self.x,x),dim=0)
			self.y = torch.cat((self.y,y),dim=0)
		else:
			self.x = x
			self.y = y

		self.fit_gp(self.x,self.y)

	def fit_gp(self,x,y, iterative = False, extrapoint = False):
		"""
		Fits the Gaussian process, possible update is via iterative inverse
		:param x: data x
		:param y: values y
		:param iterative: iterative inverse, where only last point of x is used
		:param extrapoint: iterative inverse must be allowed, x is the only addition
		:return:
		"""
		# first fit
		if (self.fit == False or iterative == False):
			if self.safe == True:
				x = self.make_safe(x)

			self.x = x
			self.y = y
			try:
				self.n,self.d = list(x.size())
			except:
				self.n, self.d = x.shape
			self.K = self.kernel(x, x) + self.s * self.s * torch.eye(self.n, dtype=torch.float64)

			self.fit = True
		else:
			# iterative inverse
			if (iterative == True):
				if extrapoint == False:
					last_point = self.x[-1,:].view(1,-1)
				else:
					last_point = x
				old_K = self.K
				old_Kinv = self.Kinv
			else:
				pass
		self.mean_std(x)
		return None

	def beta(self, delta = 1e-3, norm = 1):
		"""
		return concentration parameter given the current estimates

		:param delta: failure probability
		:param norm: norm assumption
		:return:
		"""
		beta_value = self.s * norm + \
					 torch.sqrt(2 * torch.log(1. / delta + torch.log(torch.det(self.K) / self.s ** self.n)))
		return beta_value




	def execute(self,xtest):
		if self.fit == True:
			K_star = self.kernel(self.x, xtest)
		else:
			K_star = None
		K_star_star = self.kernel(xtest,xtest)
		return (K_star,K_star_star)

	def mean_std(self, xtest, full = False, reuse = False):
		"""
		Return posterior mean and variance as tuple
		:param xtest: grid, numpy array (2D)
		:param full: Instead of just poinwise variance, full covariance can be outputed (bool)
		:return: (tensor,tensor)
		"""

		(K_star, K_star_star) = self.execute(xtest)

		if self.fit == False:
			if full == False:

				x = torch.sum(xtest,dim = 1)
				first = torch.diag(K_star_star).view(-1, 1)
				variance = first
				yvar = torch.sqrt(variance)
			else:
				x = torch.sum(xtest, dim=1)
				first = K_star_star
				yvar = first

			return ( 0*x.view(-1,1),yvar)

		if self.back_prop == False:
			if reuse == False:
				self.decomp = torch.lu(self.K.unsqueeze(0))
				self.A = torch.lu_solve(self.y.unsqueeze(0), *self.decomp)[0, :, :]
			else:
				pass
			self.B = torch.t(torch.lu_solve(torch.t(K_star).unsqueeze(0), *self.decomp)[0,:,:])
		else:
			self.A, _ = torch.lstsq(self.y,self.K)
			self.B	 = torch.t(torch.solve(torch.t(K_star),self.K)[0])

		ymean = torch.mm(K_star,self.A)

		if full == False:
			first = torch.diag(K_star_star).view(-1, 1)
			second = torch.einsum('ij,ji->i',(self.B,torch.t(K_star))).view(-1,1)
			variance = first - second
			yvar = torch.sqrt(variance)
		else:
			first = K_star_star
			second = torch.mm(self.B,torch.t(K_star))
			yvar = first - second

		return (ymean,yvar)


	def mean(self,xtest):
		K_star = self.kernel(self.x, xtest)
		ymean = torch.mm(K_star,self.A)
		return ymean

	def gradient_mean_var(self,point, hessian = True):
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
			#variance
			H = self.kernel_object.get_2_der(point)
			C = self.kernel_object.get_1_der(point,self.x)

			V = H - torch.t(C) @ self.K @ C

			return [nabla_mu,V]
		else:
			return nabla_mu

	def mean_gradient_hessian(self,xtest,hessian = False):
		hessian_mu = torch.zeros(size=(self.d,self.d), dtype = torch.float64)
		xtest.requires_grad_(True)
		#xtest.retain_grad()
		mu = self.mean_std(xtest)[0]
		#mu.backward(retain_graph=True)

		#nabla_mu = xtest.grad
		nabla_mu = grad(mu, xtest, create_graph=True)[0][0]

		if hessian == False:
			return nabla_mu
		else:
			for i in range(self.d):
				hessian_mu[i, :]= grad(nabla_mu[i], xtest, create_graph=True, retain_graph=True)[0][0]
			return [nabla_mu, hessian_mu]

	def sample(self,xtest, size = 1):
		"""
			Samples Path from GP, return a numpy array evaluated over grid
		:param xtest: grid
		:param size: number of samples
		:return: numpy array
		"""
		nn = list(xtest.size())[0]

		if self.fit == True:
			(ymean,yvar) = self.mean_std(xtest, full = True)
			Cov = yvar   + 10e-10*torch.eye(nn, dtype = torch.float64)
			L = torch.linalg.cholesky(Cov.transpose(-2, -1).conj()).transpose(-2, -1).conj()
			random_vector = torch.normal(mean=torch.zeros(nn, size, dtype = torch.float64), std=1.)
			f = ymean + torch.mm(L, random_vector)
		else:
			(K_star,K_star_star) =  self.execute(xtest)
			#L = torch.linalg.cholesky((K_star_star+ (10e-5) * torch.eye(nn, dtype=torch.float64)).transpose(-2, -1).conj()).transpose(-2, -1).conj()
			L = torch.linalg.cholesky(K_star_star+ (10e-5) * torch.eye(nn, dtype=torch.float64).double())
			random_vector = torch.normal(mean=torch.zeros(nn, size, dtype = torch.float64), std=1.)
			f = self.mu + torch.mm(L, random_vector)
		return f

	def sample_and_max(self,xtest, size = 1):
		"""
			Samples Path from GP and takes argmax
		:param xtest: grid
		:param size: number of samples
		:return: (argmax, max)
		"""
		f = self.sample(xtest,size = size)
		self.temp = f
		val, index = torch.max(f, dim = 0)
		return (xtest[index,:],val)


	def log_marginal(self,kernel,X):
		func = kernel.get_kernel()
		K = func(self.x, self.x,**X) + torch.eye(self.n, dtype=torch.float64) * self.s * self.s
		L = torch.linalg.cholesky(K)
		logdet = -0.5 * 2 * torch.sum(torch.log(torch.diag(L)))
		alpha = torch.solve(self.y,K)[0]
		logprob = -0.5 * torch.mm(torch.t(self.y),alpha) + logdet
		logprob = -logprob
		return logprob



	def optimize_params(self, type = 'bandwidth', restarts = 10, regularizer = None,
						maxiter = 1000, mingradnorm = 1e-4, verbose = False):

		# dimension of the kernel input

		# default optimizer
		optimizer = "pymanopt"

		# Spectral norm regularizer
		if regularizer is not None:
			if regularizer[0] == "spectral_norm":
				regularizer_func = lambda S: regularizer[1]*torch.norm(S[0],p='nuc')
			else:
				regularizer_func = None
		else:
			regularizer_func = None

		if type == "bandwidth":
			params = {}
			for key,dict2 in self.kernel_object.params_dict.items():
				if 'gamma' in dict2.keys():
					params[key] = {'gamma':(None, Euclidean(1), None)}
				elif 'ard_gamma' in dict2.keys():
					params[key] = {'ard_gamma': (None, Euclidean(len(dict2['group'])), None)}
		elif type == "rots":
			params = {}
			d = int(self.kernel_object.d)
			for key,dict2 in self.kernel_object.params_dict.items():
				if 'rot' in dict2.keys():
					params[key] = {'rot':(None, Stiefel(d,d), None)}
		elif type == "groups":
			params = {}
			optimizer = "discrete"
			d = self.kernel_object.d
			for key,dict2 in self.kernel_object.params_dict.items():
				if 'groups' in dict2.keys():
					params[key] = {'groups':(None, helper.generate_groups(d), None)}
			pass
		elif type == "covariance":
			params = {}
			d = int(self.kernel_object.d)
			for key,dict2 in self.kernel_object.params_dict.items():
				if 'cov' in dict2.keys():
					params[key] = {'cov':(None, PSDFixedRank(d,d), None)}
		else:
			raise AttributeError("This quick-optimization is not implemented.")


		self.optimize_params_general(params = params, restarts = restarts,
									 optimizer=optimizer, regularizer_func = regularizer_func,
									 maxiter = maxiter, mingradnorm = mingradnorm, verbose = False)


	def optimize_params_general(self,params = {}, restarts = 2,
								optimizer = "pymanopt", maxiter = 1000,
								mingradnorm = 1e-4, regularizer_func = None, verbose = False):
		"""
		:param:
		:return:
		"""
		egrad = ehess = None
		@helper.conditional_decorator(pymanopt.function.PyTorch, optimizer == "pymanopt")
		def cost(*args):
			#print (args)
			input_dict = {}
			i = 0
			for key, dict_params in params.items():
				small_param = {}
				for var_name, value in dict_params.items():
					small_param[var_name] = args[i]
					i = i + 1
				input_dict[key] = small_param

			if regularizer_func is not None:
				f = self.log_marginal(self.kernel_object,input_dict) + regularizer_func(args)
			else:
				f = self.log_marginal(self.kernel_object, input_dict)
			return f

		manifolds = []
		bounds = []
		init_values = []

		for key,dict_params in params.items():
			for var_name,value in dict_params.items():
				init_value ,manifold, bound = value
				manifolds.append(manifold)
				bounds.append(bound)
				init_values.append(init_value)


		if optimizer == "pymanopt":
			manifold = Product(tuple(manifolds))
			problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity = int(verbose)+1)
			solver = SteepestDescent(logverbosity = 1, maxiter=maxiter, mingradnorm=mingradnorm)

			# get initial point
			objective_values = []
			objective_params = []

			for _  in range(restarts):
				x_init = []
				for index,man in enumerate(manifolds):
					if init_values[index] is None:
						x_sub = man.rand()
					else:
						x_sub = np.array([init_values[index]])
					x_init.append(x_sub)
				#try:
				x_opt, log = solver.solve(problem, x = x_init)

				objective_params.append(x_opt)
				objective_values.append(log['final_values']['f(x)'])
				#except Exception as e:
				#	print (e)
				#	print ("Optimization restart failed:", x_init)
			# pick the smallest objective
			best_index = np.argmin(objective_values)
			x_opt = [torch.from_numpy(j) for j in objective_params[best_index]]

		elif optimizer == "scipy":
			cost_numpy = lambda x: cost(x).detach.numpy()
			egrad_numpy = lambda x: egrad(x).detach().numpy()
			## TODO: add a general scipy optimizer with the function

		elif optimizer == "discrete":
			values = []
			configurations = manifolds[0]
			for config in manifolds[0]:
				values.append(cost(config))

			best_index = np.argmin(values)
			x_opt = [configurations[best_index]]
		else:
			raise AssertionError("Optimizer not implemented.")


		# put back into default dic
		i = 0
		for key, dict_params in params.items():
			for var_name, value in dict_params.items():
				self.kernel_object.params_dict[key][var_name] = x_opt[i]
				i = i + 1

		#print ("--------- Finished. ------------")
		#print (self.kernel_object.params_dict)



		# disable back_prop
		self.back_prop = False

		# refit the model
		self.fit = False
		self.fit_gp(self.x, self.y)

		print(self.description())
		return True


		# def bandwidth_opt_handler():
		#
		# 	def bandwidth_opt(X):
		# 		gamma = X
		# 		return self.log_marginal_kernel_params(X)
		#
		# 	manifold = Euclidean(self.kernel_object.gamma.size()[0])
		# 	C = CostFunction(bandwidth_opt, number_args = 1)
		# 	xinit = lambda : np.random.randn()**2+np.abs(torch.zeros(self.kernel_object.gamma.size()[0], dtype = torch.float64).numpy())
		# 	return optimize(manifold, C, 1, xinit)
		#
		# # finalize
		# if type == "bandwidth":
		# 	best_params = bandwidth_opt_handler()
		# 	self.kernel_object.gamma = torch.abs(best_params[0]).detach()



		#
		# elif type == "rots":
		# 	best_params = rotations_opt_handler()
		# 	Rot = best_params[0].detach()
		# 	self.Rot = Rot
		# 	self.x = torch.mm(self.x, Rot).detach()
		#
		# elif type == "bandwidth+kappa":
		# 	best_params = bandwidth_kappa_opt_handler()
		# 	self.kernel_object.gamma = torch.abs(best_params[0]).detach()
		# 	self.s = torch.abs(best_params[1]).detach()
		#
		# elif type == "bandwidth+rots":
		# 	best_params = bandwidth_rotations_opt_handler()
		# 	self.kernel_object.gamma = torch.abs(best_params[0]).detach()
		# 	Rot = best_params[1].detach()
		# 	print("Rotation:", Rot)
		# 	self.Rot = Rot
		# 	self.x = torch.mm(self.x, Rot).detach()
		#
		# elif type == "bandwidth+kappa+rots":
		# 	best_params = bandwidth_kappa_rotations_opt_handler()
		# 	self.kernel_object.gamma = torch.abs(best_params[0]).detach()
		# 	self.s = torch.abs(best_params[1]).detach()
		# 	Rot = best_params[2].detach()
		# 	print("Rotation:", Rot)
		# 	self.Rot = Rot
		# 	self.x = torch.mm(self.x, Rot).detach()
		# elif type == "custom":
		# 	best_params = handler()
		#
		# else:
		# 	raise AttributeError("Optimization scheme not implemented")
		#


		## Bandwidth optimization

	#
	# def bandwidth_kappa_opt(X):
	# 	gamma = X[0]
	# 	kappa = X[1]
	# 	Rot = torch.eye(self.x.size()[1], dtype=torch.float64)
	# 	return self.log_marginal_likelihood(gamma,Rot,kappa, kernel=" ")
	#
	# def bandwidth_kappa_opt_handler():
	# 	manifold1 = Euclidean(self.kernel_object.gamma.size()[0])
	# 	manifold2 = Euclidean(1)
	# 	manifold = Product((manifold1, manifold2))
	# 	C = CostFunction(bandwidth_kappa_opt, number_args = 2)
	# 	xinit = lambda x: [torch.randn(self.kernel_object.gamma.size()[0], dtype = torch.float64).numpy(),np.abs(torch.randn(1,dtype = torch.float64).numpy())]
	# 	return optimize(manifold, C, 2, xinit)
	#
	#
	#
	#
	# ## Rotations optimization
	# def rotations_opt(X):
	# 	Rot = X
	# 	return self.log_marginal_likelihood(self.kernel_object.gamma,Rot, self.kernel_object.kappa, kernel=" ")
	#
	# def rotations_opt_handler():
	# 	rots = Stiefel(self.kernel_object.gamma.size()[0],self.kernel_object.gamma.size()[0])
	# 	manifold = rots
	# 	#xinit = lambda : [torch.qr(torch.randn(self.x.size()[1],self.x.size()[1], dtype=torch.float64))[0].numpy(),np.abs(torch.randn(1,dtype = torch.float64).numpy())]
	# 	xinit = lambda : torch.qr(torch.randn(self.x.size()[1], self.x.size()[1], dtype=torch.float64))[0].numpy()
	# 	C = CostFunction(rotations_opt, number_args = 1)
	# 	return optimize(manifold, C, 1, xinit)
	#
	# ## Bandwidth and Rotations optimization
	# def bandwith_rotations_opt(X):
	# 	gamma = X[0]
	# 	Rot = X[1]
	# 	return self.log_marginal_likelihood(gamma,Rot, 0.1, kernel=" ")
	#
	# def bandwidth_rotations_opt_handler():
	# 	eucl = Euclidean(self.kernel_object.gamma.size()[0])
	# 	rots = Rotations(self.kernel_object.gamma.size()[0])
	# 	manifold = Product((eucl, rots))
	# 	xinit = lambda : [torch.randn(self.kernel_object.gamma.size()[0], dtype = torch.float64).numpy(),torch.qr(torch.randn(self.x.size()[1],self.x.size()[1], dtype=torch.float64))[0].numpy()]
	# 	C = CostFunction(bandwith_rotations_opt, number_args = 2)
	# 	return optimize(manifold, C, 2, xinit)




	def log_probability(self, xtest, sample):
		from scipy.stats import multivariate_normal
		mu, covar = self.mean_std(xtest, full = True)
		p = np.log(multivariate_normal.pdf(sample.view(-1).numpy(), mean = mu.view(-1).numpy(), cov = covar.numpy()))
		return p













	def volume_mean_cvxpy(self, xtest, weights = None, eps = 10e-2,
						  tol = 10e-14, max_weight = 1, max_iter = 1000,
						  verbose = False, scale = 10e-4, slope = 1.,
						  bisections = 10, B ='auto', optimal_scale = None,
						  optimize_scale = False, relax = 'relu'):

		n = self.x.size()[0]
		K = self.get_kernel()#(self.x, self.x)
		Kinv = torch.pinverse(K+eps*torch.eye(K.size()[0]).double()).numpy()
		if weights is None:
			weights = torch.ones(self.x.size()[0])/n
		if B == 'auto':
			alpha, _ = torch.lstsq(self.y, K)
			beta = K @ alpha
			B = beta.T@Kinv@beta
			print ("Auto:B",B)
		def fun(scale_arg):
			beta = cp.Variable(n)
			if relax == 'relu':
				loss_fn_transformed = cp.sum(cp.pos(weights*slope*(cp.abs(beta - self.y.numpy().reshape(-1)) - eps))) +  0.5*scale_arg*cp.quad_form(beta, Kinv)
			elif relax == 'log':
				loss_fn_transformed = cp.sum(cp.logistic(weights * slope * (
							cp.abs(beta - self.y.numpy().reshape(-1)) - eps))) + 0.5 * scale_arg * cp.quad_form(beta,
																												Kinv)

			#loss_fn_transformed = cp.sum(weights*logit(slope*(cp.abs(beta - self.y.numpy().reshape(-1)) -eps))) +  0.5*scale_arg*cp.quad_form(beta, Kinv)-

			prob = cp.Problem(cp.Minimize( loss_fn_transformed ))
			#prob.solve(solver=cp.MOSEK, feastol=tol, verbose=False)
			prob.solve(solver=cp.MOSEK, verbose=False)
			if verbose == True:
				print ("scale:", scale_arg, "cond:", np.linalg.cond(Kinv),"sub.",beta.value.T@Kinv@beta.value - B,"B:", B)
			return beta.value.T@Kinv@beta.value - B

		if optimize_scale:
			return helper.bisection(fun, 0., max_weight, bisections)

		if optimal_scale is None:
			scale_star = helper.bisection(fun, 0., max_weight, bisections)
		else:
			scale_star = optimal_scale




		beta = cp.Variable(n)
		if relax == 'relu':
			loss_fn_transformed = cp.sum(weights * cp.pos(slope * (cp.abs(beta - self.y.numpy().reshape(-1)) - eps))) + 0.5 * scale_star * cp.quad_form(	beta, Kinv)
		elif relax == 'log':
			loss_fn_transformed = cp.sum(weights * cp.logistic(
				slope * (cp.abs(beta - self.y.numpy().reshape(-1)) - eps))) + 0.5 * scale_star * cp.quad_form(beta,
																											  Kinv)
		prob = cp.Problem(cp.Minimize(loss_fn_transformed))
		#prob.solve(solver=cp.CVXOPT, feastol=tol, verbose=verbose)
		prob.solve(solver=cp.MOSEK, verbose=verbose)
		beta_torch =  torch.from_numpy(beta.value).view(-1,1)
		alpha = torch.from_numpy(Kinv) @ beta_torch
		ytest = self.kernel(self.x, xtest)  @ alpha
		return ytest


	def volume_mean(self, xtest, weights = None, eps = 10e-2, tol = 10e-6, max_iter = 1000, verbose = False, eta_start = 0.01,
					eta_decrease =0.9, scale = 1, slope = 1., warm = True, relax = 'relu', norm = False, B = 'auto'):
		self.scale = scale
		self.relax = relax


		K = self.get_kernel()#(self.x, self.x)
		Kinv = torch.pinverse(K)

		if weights is None:
			weights = torch.ones(self.x.size()[0])
		else:
			weights[weights<10e-6] = 0.#* self.x.size()[0]
			weights = weights.view(-1)
		if warm==True:
			# warm start with L2 fit
			alpha, _ = torch.lstsq(self.y,K)
			beta = K @ alpha
		else:
			beta = torch.randn(size = (self.n, 1)).double()#.requires_grad_(True)*0

		#loss_fn_original = lambda alpha: torch.sum(torch.relu(torch.abs(K @ alpha - self.y) -eps)) + 0.5*self.s * alpha.T @ K @ alpha
		if self.relax == "relu":
			loss_fn_transformed = lambda beta: torch.sum(torch.relu(torch.abs(beta - self.y) -eps)) + self.scale*0.5*self.s * beta.T @ Kinv @ beta

		elif self.relax == "tanh":
			self.slope = slope
			tanh = lambda x: (torch.tanh(self.slope * x) + 1) * 0.5
			loss_fn_transformed = lambda beta: torch.sum(weights*tanh(torch.abs(beta - self.y) -eps).view(-1)) + 0.5*self.s*self.scale*beta.T @ Kinv @ beta

		elif self.relax == "elu":
			self.slope = slope
			elu = lambda x: torch.nn.elu(x, alpha = self.slope)
			loss_fn_transformed = lambda beta: torch.sum(elu(torch.abs(beta - self.y) -eps)) + 0.5*self.s *self.scale*beta.T @ Kinv @ beta

		elif self.relax == "relu":
			return self.volume_mean_cvxpy(xtest,weights = weights, eps = eps, scale = scale, tol = tol)
		else:
			raise AssertionError("Unkown relaxation.")



		current_loss = 10e10
		eta = eta_start
		for i in range(max_iter):
			grad = self.s * (Kinv @ beta)
			beta = self.proximal(beta,grad,eta,eps,weights)
			past_loss = current_loss
			current_loss = loss_fn_transformed(beta)
			if current_loss>past_loss:
				eta = eta*eta_decrease
			elif np.abs(current_loss-past_loss)<tol:
				break

			#print (i, beta.T)
			if verbose == True:
				print (i, loss_fn_transformed(beta),eta)

		print ("final norm:",beta.T@Kinv@beta)

		#alpha = torch.inverse(self.K) @ beta
		alpha = torch.pinverse(K) @ beta
		#alpha = torch.lstsq(K,beta)
		ytest = self.kernel(self.x, xtest)  @ alpha
		# max = torch.max(torch.abs(beta - self.y))
		if norm == True:
			return beta.T@Kinv@beta
		#yz = self.kernel(self.x, self.x)  @ alpha
		# approx_v = torch.sum(torch.relu(torch.abs(beta - self.y) -eps))/max
		# approx_p = approx_v/self.n
		#mask = (torch.abs(yz[:,0] - self.y[:,0])) > eps
		#approx_p = float(torch.sum(mask))/float(self.n)
		return ytest#,approx_p


	def volume_mean_norm(self, xtest, weights = None, eps = 10e-2, tol = 10e-6, max_iter = 1000, verbose = False, eta_start = 0.01,
					eta_decrease =0.9, scale = 1, slope = 1., warm = True, relax = 'relu', B = 'auto'):
		K = self.kernel(self.x, self.x)
		Kinv = torch.pinverse(K)
		if B == 'auto':
			alpha, _ = torch.lstsq(self.y, self.K)
			beta = K @ alpha
			B = beta.T@Kinv@beta

		func = lambda s : self.volume_mean(xtest, weights=weights, eps=eps, tol=tol, max_iter=max_iter, verbose=verbose, eta_start=eta_start,
					eta_decrease=eta_decrease, scale=s, slope=slope, warm=warm, relax=relax, norm = True) - B

		s_star = stpy.optim.custom_optimizers.bisection(func,0.,1000.,10)

		return self.volume_mean(xtest, weights=weights, eps=eps, tol=tol, max_iter=max_iter, verbose=verbose,
						 eta_start=eta_start,
						 eta_decrease=eta_decrease, scale=s_star,	slope = slope, warm = warm, relax = relax, norm = False)


	def proximal(self,beta,nabla,eta, eps, weights):
		res = beta
		for i in range(self.n):
			from scipy.optimize import minimize

			b = float(beta[i,:])
			y = float(self.y[i,:])
			g = float(nabla[i,:])
			w = float(weights[i])
			#s = float(self.s)

			tanh = lambda x: (np.tanh(self.slope*x)+1)*0.5
			elu = lambda x: torch.elu(x,alpha = self.slope).numpy()

			if self.relax == "relu":
				loss_reg = lambda x: w*np.maximum(0,np.abs(x-y)-eps)
			elif self.relax == "tanh":
				loss_reg = lambda x: w*tanh(np.abs(x-y)-eps)
			elif self.relax == "elu":
				loss_reg = lambda x: w*elu(np.abs(x-y)-eps)
			else:
				raise AssertionError("Unkown relaxation.")

			loss_scalar = lambda x: ((1 / (2. * eta)) * (x - (b - eta*g)) ** 2) + loss_reg(x)

			x0 = np.array([0.])
			#print (minimize(loss_scalar,x0,method ='nelder-mead').x)
			res[i,:] = float(minimize(loss_scalar,x0,method ='nelder-mead').x)
		return res




	def get_lambdas(self,beta, mean = False):
		"""
		Gets lambda function to evaluate acquisiton function and its derivative
		:param beta: beta in GP-UCB
		:return: [lambda,lambda]
		"""
		mean = lambda x : self.mean_std(x.reshape(1, -1), reuse = True)[0][0][0]
		sigma = lambda x : self.mean_std(x.reshape(1, -1), reuse = True)[1][0][0]

		if mean == True:
			return [mean,sigma]
		else:
			fun = lambda x: -(mean(x) + np.sqrt(beta)*sigma(x))
			grad = lambda x: -complex_step_derivative(fun,1e-10,x.reshape(1,-1))

			return [fun, grad]

	def get_kernel(self):
		return self.K

	def ucb_optimize(self,beta, multistart = 25, lcb = False):
		"""
		Optimizes UCB acquisiton function and return next point and its value as output
		:param beta: beta from GP UCB
		:param multistart: number of starts
		:return: (next_point, value at next_point)
		"""

		mean = lambda x: self.mean_std(x, reuse = True)[0][0][0]
		sigma = lambda x: self.mean_std(x, reuse = True)[1][0][0]

		ucb = lambda x: torch.dot(torch.Tensor([1.0,np.sqrt(beta)]), torch.Tensor([self.mean_std(x, reuse = True)[0][0][0], self.mean_std(x, reuse = True)[1][0][0]]))
		lcb = lambda x: torch.dot(torch.Tensor([1.0,np.sqrt(beta)]), torch.Tensor([self.mean_std(x, reuse = True)[0][0][0], -self.mean_std(x, reuse = True)[1][0][0]]))

		if lcb == False:
			fun2 = lambda x: -ucb(torch.from_numpy(x).view(1,-1)).numpy()
		else:
			fun2 = lambda x: -lcb(torch.from_numpy(x).view(1, -1)).numpy()
		fun = lambda x: -(mean(torch.from_numpy(x).view(1, -1)) + np.sqrt(beta) * sigma(torch.from_numpy(x).view(1, -1)))

		self.back_prop = False
		self.mean_std(self.x)

		mybounds = self.bounds

		results = []

		from scipy.optimize import minimize,fmin_tnc
		
		for i in range(multistart):
			x0 = np.random.randn(self.d)
			for i in range(self.d):
				x0[i] = np.random.uniform(mybounds[i][0],mybounds[i][1])

			res = minimize(fun2, x0, method = "L-BFGS-B", jac = None, tol = 0.000001, bounds=mybounds)
			solution = res.x
			results.append([solution,-fun(solution)])

		results = np.array(results)
		index = np.argmax(results[:,1])
		solution = results[index,0]

		return (torch.from_numpy(solution),-fun(solution))



	def isin(self, xnext):
		self.epsilon = 0.001
		for v in self.x:
			if torch.norm(v - xnext,p =2) < self.epsilon:
				return True

	def sample_and_condition(self,x):
		xprobe = x.view(1,-1)
		fprobe = self.sample(xprobe)
		if not self.isin(xprobe):
			self.x = torch.cat((self.x, xprobe), dim=0)
			self.y = torch.cat((self.y, fprobe), dim=0)
			self.fit_gp(self.x,self.y)
		return -fprobe

	def get_lambdas_TH(self):
		fun = lambda x: self.sample_and_condition(x)
		grad = None
		return [fun,grad]


	def sample_iteratively_max(self,xtest, multistart = 20, minimizer = "coordinate-wise", grid = 100):
		"""
			Samples Path from GP and takes the maximum iteratively
			:param xtest: grid
			:param size: number of samples
			:return: numpy array
		"""
		#print ("Iterative:",multistart,minimizer,grid)
		from scipy.optimize import minimize, fmin_tnc
		# old stuff
		xold = self.x
		yold = self.y

		# with fixed grid
		if xtest is not None:
			# number of samples
			nn = xtest.shape[0]

			f = torch.zeros(nn,dtype = torch.float64)

			for j in range(nn):
				xprobe = xtest[j,:].view(1,-1)
				(K_star, K_star_star) = self.execute(xprobe)
				(ymean, yvar) = self.mean_std(xprobe)
				L = torch.sqrt(K_star_star + self.s * self.s * torch.eye(1,dtype = torch.float64) - yvar)
				fprobe = ymean + L*torch.randn(1,dtype = torch.float64)
				# add x and fprobe to the dataset and redo the whole
				f[j] = fprobe
				if not self.isin(xprobe):
					self.x = torch.cat((self.x, xprobe), dim=0)
					self.y = torch.cat((self.y, fprobe), dim=0)

				self.fit_gp(self.x,self.y)

			val, index = torch.max(f, dim=0)
			self.fit_gp(xold,yold)
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

				#print ("Multistart:",j)
				x0 = torch.randn(self.d, dtype = torch.float64)
				for i in range(self.d):
					x0[i].uniform_(mybounds[i][0], mybounds[i][1])

				# simple coordnate-wise optimization
				if minimizer == "coordinate-wise":
					solution = x0
					for i in range(self.d):
						xtest = torch.from_numpy(np.tile(x0, (grid,1)))
						xtest[:,i] = torch.linspace(mybounds[i][0],mybounds[i][1],grid)
						sample = self.sample(xtest)

						## Add to the posterior
						self.x = torch.cat((self.x, xtest), dim=0)
						self.y = torch.cat((self.y, sample), dim=0)

						# argmax
						val, index = torch.max(sample, dim=0)
						out = xtest[index, :]

						#fit new GP
						self.fit_gp(self.x, self.y)
						solution[i] = out[0,i]

				elif minimizer == "L-BFGS-B":
					solution = np.random.randn(self.d)
					xmax = [b[1] for b in mybounds]
					xmin = [b[0] for b in mybounds]
					bounds = MyBounds(xmax=xmax,xmin=xmin)
					func = lambda x: fun(torch.from_numpy(x)).numpy()[0][0]
					res = scipy.optimize.basinhopping(func,solution, disp = False, niter = grid, accept_test=bounds)
					solution = torch.from_numpy(res.x)

				else:
					raise AssertionError("Wrong optimizer selected.")

				results.append(torch.cat((solution, -fun(solution)[0])))
				self.x = xold
				self.y = yold
				self.fit_gp(self.x, self.y)

			results = torch.stack(results)
			val, index = torch.max(results[:, -1], dim = 0)
			solution = results[index, 0:self.d].view(1,self.d)
			self.x = xold
			self.y = yold
			self.fit_gp(self.x, self.y)

			return (solution, -fun(solution))





if __name__=="__main__":
	# domain size
	L_infinity_ball = 1
	# dimension
	d = 1
	# error variance
	s = torch.from_numpy(np.array(1.0,dtype = np.float64))

	# grid density
	n = 1024
	# number of intial points
	N = 32
	# smoothness
	gamma = 0.1
	# test problem

	xtest = torch.from_numpy(interval(n,d))
	#x = torch.from_numpy(np.random.uniform(-L_infinity_ball,L_infinity_ball, size = (N,d)))
	x = torch.from_numpy(interval(N,1))
	f_no_noise = lambda q: torch.sin(torch.sum(q*4, dim = 1)).view(-1,1)
	f = lambda q: f_no_noise(q) + torch.normal(mean=torch.zeros(q.size()[0],1,dtype = torch.float64), std = 1., out=None)*s*s
	# targets
	y = f(x)

	# GP model with squared exponential
	#kernel = KernelFunction(kernel_name = "ard", gamma = torch.ones(d, dtype = torch.float64)*gamma , groups = [[0],[1]])
	#kernel = KernelFunction(kernel_name="ard", gamma=torch.ones(1, dtype=torch.float64) * gamma, groups=[[0]])
	GP = GaussianProcess(s = s,d = 1, gamma = gamma)

	# fit GP
	#x = x.numpy()
	GP.fit_gp(x,y)
	# get mean and variance of GP
	[mu,std] = GP.mean_std(xtest)


	#print ("Log probability:", GP.log_marginal_likelihood() )
	#mu_inf = GP.chebyshev_mean(xtest)
	eps = 0.1

	mu_vol = GP.volume_mean_cvxpy(xtest, eps=eps, verbose=True, scale=1., slope=1., tol=10e-9)

	GP.visualize(xtest, f_true = f_no_noise, show = False)
	plt.plot(xtest.numpy(),mu_vol.detach().numpy(), label = "Least-Volume-ReLu", lw = 2)
	for slope in [0.001,0.01,0.1,1.,10.,100.,1000.,10000.]:
		#mu_vol_log = GP.volume_mean_cvxpy(xtest, eps=eps, verbose=True, scale=1., slope=slope, tol=10e-9, relax = 'log', B = 1000)
		#plt.plot(xtest.numpy(),mu_vol_log.detach().numpy(), '--',label = "Least-Volume-Log" + str(slope), lw = 2)
		mu_vol_tanh = GP.volume_mean(xtest, eps=eps, verbose=True, eta_start=0.1, eta_decrease=0.1, scale=1., slope=slope,
									 tol=0.01, warm=True, relax='tanh')
		plt.plot(xtest.numpy(), mu_vol_tanh.detach().numpy(), '-.', label="Least-Volume-Tanh" + str(slope), lw=2)
	#print (slope, np.sum(np.abs(mu_vol_log)<eps) )
	#plt.plot(xtest.numpy(),mu_vol_tanh.detach().numpy(), label = "Least-Volume-Tahn", lw = 2)
	#plt.plot(xtest.numpy(),mu_vol_tanh2.detach().numpy(), label = "Least-Volume-Tahn2", lw = 2)

	#plt.plot(xtest.numpy(),mu_inf.detach().numpy(), label = "Chebyschev estimate", lw = 2)
	plt.plot(x.numpy(), y.numpy() + eps,'ko')
	plt.plot(x.numpy(), y.numpy() - eps,'ko')
	plt.legend()
	plt.show()