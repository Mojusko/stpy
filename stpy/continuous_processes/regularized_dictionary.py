from typing import Union, Tuple, List
from stpy.optim.custom_optimizers import bisection
import cvxpy as cp
import mosek
import numpy as np
import torch
from scipy.optimize import minimize

from stpy.embeddings.embedding import Embedding
from stpy.estimator import Estimator
from stpy.helpers.constraints import Constraints
from stpy.probability.likelihood import Likelihood
from stpy.probability.regularizer import L2Regularizer, Regularizer


class RegularizedDictionary(Estimator):
	"""
		meta-class for regularized dictionary estimator
	"""

	def __init__(self,
				 embedding: Embedding,
				 likelihood: Likelihood,
				 regularizer: Regularizer = L2Regularizer(lam = 1.),
				 constraints: Union[Constraints,None] = None,
				 use_constraint: bool = False,
				 d: int = 1,
				 verbose: bool=True,
				 groups: bool =None,
				 bounds: bool =None):

		self.likelihood = likelihood
		self.embedding = embedding
		self.regularizer = regularizer
		self.constraints = constraints
		self.use_constraint = use_constraint

		self.m = self.embedding.get_m()
		self.x = None

		self.fitted = False
		self.data = False

		self.d = d
		self.n = 0
		self.bounds = bounds
		self.groups = groups
		self.verbose = verbose

		self.admits_first_order = True
		self.to_add = []

	def description(self):
		return "regularized dictionary object"

	def embed(self, x):
		return self.embedding.embed(x)


	def similarity_kernel(self, x, y):
		embeddingx = self.embed(x)
		embeddingy = self.embed(y)
		return self.linear_kernel(embeddingx, embeddingy)

	def effective_dimension(self, xtest, cuttoff = None):
		if cuttoff is None:
			cuttoff = self.lam
		Phi = self.embed(xtest)
		d_eff = torch.trace(torch.solve(Phi.T @ Phi, Phi.T @ Phi + torch.eye(self.m).double() * cuttoff )[0])
		return d_eff

	def add_points(self, x, y):
		if self.x is not None:
			self.x = torch.cat((self.x, x), dim=0)
			self.y = torch.cat((self.y, y), dim=0)
		else:
			self.x = x
			self.y = y

	def add_data_point(self, structured_datapoint: Union[Tuple, List]):
		if self.n == 0:
			self.load_data(structured_datapoint)
		else:
			self.to_add.append(structured_datapoint)
			self.fitted = False

	def load_data(self, data: Union[Tuple, List]):
		"""

		:param data:
		:return:
		"""
		x, y = data
		self.x = x
		self.y = y
		self.n = list(self.x.size())[0]
		self.d = list(self.x.size())[1]
		self.data = True
		self.fitted = False

	def fit(self):
		data = (self.embed(self.x), self.y)
		self.likelihood.load_data(data)

	def calculate_non_convex(self, repeats = 100):
		if self.regularizer.groups is None:
			eta = np.ones(self.m)*1
		else:
			eta = np.ones(len(self.regularizer.groups))*1

		for _ in range(repeats):
			theta = cp.Variable((self.m, 1))
			likelihood = self.likelihood.get_cvxpy_objective()
			regularizer = self.regularizer.get_cvxpy_regularizer(eta)
			objective = likelihood(theta) + regularizer(theta)
			constraints = []
			if self.constraints is not None and self.use_constraint:
				set = self.constraints.get_cvxpy_constraint(theta)
				constraints += set

			prob = cp.Problem(cp.Minimize(objective), constraints)
			prob.solve(solver = cp.MOSEK, mosek_params={ mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})
			if self.regularizer.groups is None:
				eta = np.abs(theta.value)**(2-self.regularizer.q)
			else:
				eta = np.array([np.linalg.norm(theta.value[group]) ** (2 - self.regularizer.q) for group in self.regularizer.groups])
			eta = eta + 1e-8

			#print (theta.value)
		self.theta_fit = torch.from_numpy(theta.value)
		self.fitted = True
		return theta.value

	def calculate(self):

		if self.fitted:
			pass
		elif self.regularizer.is_convex():
			theta = cp.Variable((self.m,1))
			likelihood = self.likelihood.get_cvxpy_objective()
			regularizer = self.regularizer.get_cvxpy_regularizer()
			objective = likelihood(theta) + regularizer(theta)

			constraints = []
			if self.constraints is not None and self.use_constraint:
				set = self.constraints.get_cvxpy_constraint(theta)
				constraints += set

			prob = cp.Problem(cp.Minimize(objective), constraints)
			prob.solve(solver = cp.MOSEK, mosek_params={ mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})
			self.theta_fit = torch.from_numpy(theta.value)
			self.fitted = True
		else:
			self.calculate_non_convex()

	def mean_parameter(self):
		## TODO: add here the mean
		theta = None
		return theta

	def objective_on_confidence_set(self, theta, objective, type = None):

		set = self.likelihood.get_cvxpy_confidence_set(theta, type = type,
												information = [self.theta_fit,
												 self.regularizer.hessian(self.theta_fit)])

		constraints = []
		constraints += set

		if self.constraints is not None:
			constraint = self.constraints.get_cvxpy_constraint(theta)
			constraints += constraint

		prob = cp.Problem(cp.Minimize(objective), constraints)
		prob.solve(solver = cp.MOSEK, mosek_params={ mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})

		value =  prob.value
		return value, theta.value

	def objective_on_non_convex_confidence_set_bisection(self, theta, objective, type = None):

		def optimize_for_lam(lam, self, objective, theta):

			if self.regularizer.groups is None:
				eta = np.ones(self.m) * 1
			else:
				eta = np.ones(len(self.regularizer.groups)) * 1
			repeats = 3

			for _ in range(repeats):
				#theta = cp.Variable((self.m, 1))
				set = self.likelihood.get_cvxpy_confidence_set(theta, type=type,
															   information=[self.theta_fit,
																			self.regularizer.hessian(self.theta_fit)])
				regularizer = self.regularizer.get_cvxpy_regularizer(eta)
				objective = objective + lam*(regularizer(theta)-1)
				constraints = set


				prob = cp.Problem(cp.Minimize(objective), constraints)
				prob.solve(solver=cp.MOSEK, mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
														  mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
														  mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
														  mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})
				if self.regularizer.groups is None:
					eta = np.abs(theta.value) ** (2 - self.regularizer.q)
				else:
					eta = np.array([np.linalg.norm(theta.value[group]) ** (2 - self.regularizer.q) for group in
									self.regularizer.groups])
				eta = eta + 1e-8
				print (eta)
			value = prob.value
			return value


		optimize_for_lam_small = lambda a : optimize_for_lam(a,self,objective,theta)
		# lam_final = bisection(optimize_for_lam_small,1e-5,10,100)
		return optimize_for_lam_small(1.0), None

	def ucb(self, xtest: torch.Tensor, type = None, arg = False):
		return self.lcb(xtest, type = type, arg = arg, sign = -1.)

	def lcb(self, xtest: torch.Tensor, type = None, arg = False, sign = 1.):
		theta = cp.Variable((self.m, 1))
		args = []
		n = xtest.size()[0]
		values = torch.zeros(size = (n,1)).double()
		Phi = self.embed(xtest)

		for j in range(n):
			objective = sign * Phi[j,:]@theta
			if not self.regularizer.is_convex():
				value, theta_lcb = self.objective_on_non_convex_confidence_set_bisection(theta, objective, type=type)
			else:
				value, theta_lcb = self.objective_on_confidence_set(theta, objective, type= type)
			values[j] = sign* value
			if arg:
				args.append(theta_lcb)

		if args:
			return values, args
		else:
			return values


	def theta_ml(self):
		self.calculate()
		return self.theta_fit

	def theta_covar(self):
		"""
		Calculate the covariance function at the current theta estimate
		:return:
		"""
		return self.likelihood.information_matrix(self.theta_fit) + self.regularize.hessian(self.theta_fit)

	def mean(self, xtest: torch.Tensor):
		'''
			Calculate mean and variance for GP at xtest points
		'''
		embeding = self.embed(xtest)

		# mean
		theta_mean = self.theta_ml()
		ymean = embeding @ theta_mean
		return ymean

	def mean_std(self, xtest: torch.Tensor):
		ymean = self.mean(xtest)
		ystd = ymean*0

		return (ymean, ystd)

	#
	# def optimize_bound(self, beta = lambda : 2, multistart=25, lcb=False, minimizer="L-BFGS-B"):
	#
	# 	if lcb:
	# 		fun = lambda x: - self.lcb(torch.from_numpy(x).view(1,-1), beta = beta).detach().numpy()[0]
	# 	else:
	# 		fun = lambda x: - self.ucb(torch.from_numpy(x).view(1,-1), beta = beta).detach().numpy()[0]
	#
	# 	if self.bounds == None:
	# 		mybounds = tuple([(-1., 1) for _ in range(self.d)])
	#
	# 	else:
	# 		mybounds = self.bounds
	#
	# 	results = []
	# 	for j in range(multistart):
	# 		x0 = np.random.randn(self.d)
	# 		for i in range(self.d):
	# 			x0[i] = np.random.uniform(mybounds[i][0], mybounds[i][1])
	#
	# 		if minimizer == "L-BFGS-B":
	# 			res = minimize(fun, x0, method="L-BFGS-B", jac=None, tol=0.0001, bounds=mybounds)
	# 			solution = res.x
	# 		else:
	# 			raise AssertionError("Wrong optimizer selected.")
	#
	# 		results.append([solution, -fun(solution)])
	#
	# 	results = np.array(results)
	# 	index = np.argmax(results[:, 1])
	# 	solution = results[index, 0]
	# 	return (torch.from_numpy(solution).view(1, -1), -torch.from_numpy(fun(solution)))
	#

