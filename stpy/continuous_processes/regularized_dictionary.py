from typing import Union, Tuple, List
import warnings
import cvxpy as cp
import mosek
import numpy as np
import torch

from stpy.embeddings.embedding import Embedding
from stpy.estimator import Estimator
from stpy.regularization.constraints import Constraints
from stpy.probability.likelihood import Likelihood
from stpy.regularization.regularizer import L2Regularizer, Regularizer
from stpy.kernels import KernelFunction

class RegularizedDictionary(Estimator):
	"""
		Main class for regularized dictionary estimators
	"""
	def __init__(self,
				 embedding: Embedding, # embedding
				 likelihood: Likelihood, # likelihood function
				 regularizer: Regularizer = L2Regularizer(lam = 1.), # regularizer
				 constraints: Union[Constraints,None] = None, # constraint object
				 use_constraint: bool = False, # whether the use the constraint in the estimation
				 d: int = 1, # dimension of the action set
				 verbose: bool=False,
				 groups: bool =None, # groups TODO: obsolete?
				 bounds: bool =None, # bound TODO: obsolete?
				 inference_type: Union [str,None] = None, # inference type; what confidence sets to use
				 accuracy: float = 1e-8, # span check accuracy
				 LR_scaling: float = 1., # scaling between signal and noise
				 check: str = None, # type of check when dealing with likelihood/bias error compensation
				 bound: float = 1., # absolute norm bound on the estimated value, the norm class is specified by the inference type
				 tolerance: float = 1e-6,  # tolerance for optimizer
				 delta: float = 0.1):


		self.likelihood = likelihood
		self.embedding = embedding
		self.regularizer = regularizer
		self.LR_scaling = LR_scaling
		self.constraints = constraints
		self.use_constraint = use_constraint
		self.check = check
		self.m = self.embedding.get_m()
		self.x = None
		self.delta = delta
		self.fitted = False
		self.data = False
		self.bound = bound
		self.d = d
		self.n = 0
		self.bounds = bounds
		self.groups = groups
		self.verbose = verbose
		self.inference_type = inference_type
		self.admits_first_order = True
		self.to_add = []
		self.d_eff = None
		self.accuracy = accuracy
		self.tolerance = tolerance
		self.evidence = []
		self.estimator_sequence = []
		self.vovk_estimator_sequence = []

	def description(self):
		return "regularized dictionary object"

	def embed(self, x):
		return self.embedding.embed(x)

	def similarity_kernel(self, x, y):
		embeddingx = self.embed(x)
		embeddingy = self.embed(y)
		return self.linear_kernel(embeddingx, embeddingy)

	def effective_dimension(self, xtest, lamH = None):
		if lamH is None:
			lamH = self.regularizer.lam
			lamH = lamH * torch.eye(xtest.size()[0]).double()
		else:
			pass
		Phi = self.embed(xtest)
		d_eff = torch.trace(torch.linalg.inv(Phi @ Phi.T + lamH)@(Phi @ Phi.T))
		return d_eff

	def set_effectitve_dimension(self, xtest):
		self.d_eff = self.effective_dimension(xtest)

	def load_data(self, data: Union[Tuple, List]):
		"""

		:param data:
		:return:
		"""
		x, y = data
		self.phi = self.embed(x)
		self.x = x
		self.y = y
		self.n = list(self.x.size())[0]
		self.d = list(self.x.size())[1]
		self.data = True
		self.fitted = False

		for i in range(self.n):

			if self.check == "bias":
				self.evidence.append(self.signal_to_noise_ratio(self.x))

			elif self.check == "span":
				self.evidence.append(False)

			elif self.check == "none":
				self.evidence.append(1.)

			self.estimator_sequence.append(torch.zeros(size = (self.m,1)).double())

	def signal_to_noise_ratio(self, x):
		bias = self.bias(x)
		print ("bias", bias)
		SNR = self.likelihood.scale(err=bias, bound = self.bound)*self.LR_scaling/(self.likelihood.scale(err=bias,bound = self.bound)*self.LR_scaling + self.bias(x))
		print ("scale",self.likelihood.scale(err=bias, bound = self.bound))
		print ("SNR", SNR)
		return float(SNR)

	def add_points(self, structured_datapoints):
		"""
		Adding more than one point

		:param structured_datapoint:
		:return:
		"""
		x,y = structured_datapoints

		if self.x is not None:

			if self.inference_type == "LR":

				if self.check == "span":
					self.evidence.append(self.span_check(x.view(1, -1)))
				elif self.check == "bias":
					self.evidence.append(self.signal_to_noise_ratio(x.view(1, -1)))
				elif self.check == "none":
					self.evidence.append(1.)

			self.x = torch.cat((self.x, x), dim=0)
			self.y = torch.cat((self.y, y), dim=0)
			self.phi = torch.cat((self.phi, self.embed(x)), dim=0)

		else:
			self.x = x
			self.y = y
			self.phi = self.embed(x)

		self.fitted = False

	def fit(self):
		data = (self.phi, self.y)
		self.likelihood.load_data(data)
		self.calculate()


	def calculate(self):

		if self.fitted:
			if self.verbose:
				print ("Skip fitting.")
		elif (self.regularizer is None or self.regularizer.is_convex()) and (self.constraints is None or self.constraints.is_convex()):
			theta = cp.Variable((self.m,1))
			likelihood = self.likelihood.get_objective_cvxpy()
			objective = likelihood(theta)

			if self.regularizer is not None:
				regularizer = self.regularizer.get_regularizer_cvxpy()
				objective += regularizer(theta)

			constraints = []
			if self.constraints is not None and self.use_constraint:
				set = self.constraints.get_constraint_cvxpy(theta)
				constraints += set

			prob = cp.Problem(cp.Minimize(objective), constraints)
			prob.solve(solver = cp.MOSEK, mosek_params={
									 mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
									 mosek.dparam.intpnt_co_tol_pfeas: self.tolerance ,
									 mosek.dparam.intpnt_co_tol_dfeas: self.tolerance ,
									 mosek.dparam.intpnt_co_tol_rel_gap: self.tolerance }, verbose = False)

			self.theta_fit = torch.from_numpy(theta.value)
			self.fitted = True

			## TODO: do this with a decorator
			if self.inference_type == "LR":
				self.update_lr_sequence()
		else:
			raise ValueError("The regularizer or constraint specified are non-convex, use a dedicated class for non-convex estimation.")



	def update_lr_sequence(self):
		if self.inference_type == "LR":
			self.estimator_sequence.append(self.theta_fit)

	def span_check(self, x):
		if self.verbose:
			print ("-----------")
			print ("Span check", x)
			print (self.x.T)

		phi = self.embed(x)
		I = torch.eye(self.m).double()
		all_phi = self.phi
		projection_matrix = I - all_phi.T @ torch.linalg.pinv(all_phi@all_phi.T) @ all_phi
		err = projection_matrix@phi.T
		err = np.sqrt(torch.sum(err**2))

		if err < self.accuracy:
			return True # its in the span
		else:
			return False # its not in the span

	def bias(self, x):
		"""
		This calculates bias squared

		:param x:
		:return:
		"""
		if self.verbose:
			print ("-----------")
			print ("Bias check", x)
			print (self.x.T)

		phi = self.embed(x)
		I = torch.eye(self.m).double()
		all_phi = self.phi
		projection_matrix = I - all_phi.T @ torch.linalg.pinv(all_phi@all_phi.T) @ all_phi
		err = projection_matrix@phi.T
		err = float(torch.sum(err**2))*self.bound**2
		return err

	def objective_on_confidence_set(self, theta, objective, inference_type = None):
		params = { 'estimate':self.theta_fit,
						 	'regularizer_hessian':self.regularizer.hessian(self.theta_fit),
						    'd_eff':self.d_eff if self.d_eff  is not None else self.m,
							'bound':self.bound,
							'kernel_object':KernelFunction(d = self.d, kernel_function=lambda x,y, kappa, group: x.T@y),
							'evidence': self.evidence,
							'estimator_sequence': self.estimator_sequence
							}

		set = self.likelihood.get_confidence_set_cvxpy(theta, type = inference_type,
													   params = params)

		constraints = []
		constraints += set

		if self.constraints is not None:
			constraint = self.constraints.get_constraint_cvxpy(theta)
			constraints += constraint

		prob = cp.Problem(cp.Minimize(objective), constraints)
		prob.solve(solver = cp.MOSEK, mosek_params={ mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-8}, verbose = False)
		value =  prob.value
		return value, theta.value

	def ucb(self, xtest: torch.Tensor):
		"""
		Calculates ucb
		:param xtest: anchor points
		:return: torch
		"""
		return self.lcb(xtest, sign = -1.)


	def get_param_conf_set(self, delta = 0.1):
		if self.regularizer is not None:
			H = self.regularizer.hessian(self.theta_fit)
		else:
			H = None
		params = { 'estimate':self.theta_fit,
						 	'regularizer_hessian':H,
						    'd_eff':self.d_eff if self.d_eff  is not None else self.m,
							'bound':self.bound,
							'kernel_object':KernelFunction(d = self.d, kernel_function=lambda x,y, kappa, group: x.T@y),
							'evidence': self.evidence,
							'estimator_sequence': self.estimator_sequence
							}
		fn = self.likelihood.get_objective_torch()
		beta = self.likelihood.confidence_parameter_likelihood_ratio(delta, params)
		return fn, beta

	def lcb(self, xtest: torch.Tensor, sign: float  = 1.):
		"""
		Calculates lcb
		:param xtest: anchor points
		:param sign: controls whether lcb or ucb is calculated
		:return: torch
		"""
		n = xtest.size()[0]
		values = torch.zeros(size=(n, 1)).double()
		Phi = self.embed(xtest)

		theta = cp.Variable((self.m, 1))
		v = cp.Parameter((self.m,1))
		objective = v.T @ theta

		if self.regularizer is not None:
			H = self.regularizer.hessian(self.theta_fit)
		else:
			H = None
		params = { 'estimate':self.theta_fit,
						 	'regularizer_hessian':H,
						    'd_eff':self.d_eff if self.d_eff  is not None else self.m,
							'bound':self.bound,
							'kernel_object':KernelFunction(d = self.d, kernel_function=lambda x,y, kappa, group: x.T@y),
							'evidence': self.evidence,
							'estimator_sequence': self.estimator_sequence
							}

		set = self.likelihood.get_confidence_set_cvxpy(theta, type = self.inference_type,
													   params = params, delta =self.delta)

		constraints = []
		constraints += set

		if self.constraints is not None:
			constraint = self.constraints.get_constraint_cvxpy(theta)
			constraints += constraint
		prob = cp.Problem(cp.Minimize(objective), constraints)

		for j in range(n):
			v.value = sign*Phi[j,:].view(-1,1).numpy()
			prob.solve(warm_start = True, solver=cp.MOSEK, mosek_params={
													mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
													mosek.dparam.intpnt_co_tol_pfeas: self.tolerance ,
			 										  mosek.dparam.intpnt_co_tol_dfeas: self.tolerance ,
			 										  mosek.dparam.intpnt_co_tol_rel_gap: self.tolerance }, verbose= False)
			values[j] = prob.value
		return sign*values

	def theta_ml(self):
		"""
		Calculates and output the parameter vector
		:return:
		"""
		self.calculate()
		return self.theta_fit

	def theta_covar(self):
		"""
		Calculate the covariance function at the current theta estimate
		:return:
		"""
		return self.likelihood.information_matrix(self.theta_fit) + self.regularizer.hessian(self.theta_fit)

	def mean(self, xtest: torch.Tensor):
		"""
		Calculate mean and variance for GP at xtest points
		"""
		embeding = self.embed(xtest)
		# mean
		theta_mean = self.theta_ml()
		ymean = embeding @ theta_mean
		return ymean

	def mean_std(self, xtest: torch.Tensor):
		"""
		Calculates map and uncertainty prediction
		:param xtest:
		:return:
		"""
		warnings.warn("This is only a legacy function, it does not provide uncertainty for the class"+str(__class__.__name__))

		ymean = self.mean(xtest)
		ystd = ymean*0
		return (ymean, ystd)


