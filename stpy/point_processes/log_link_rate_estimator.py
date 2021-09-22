import numpy as np
import scipy
import torch
import mosek
import cvxpy as cp
from stpy.helpers.quadrature_helper import quadvec2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from stpy.embeddings.embedding import HermiteEmbedding
import scipy.integrate as integrate
from stpy.helpers.ellipsoid_algorithms import maximize_quadratic_on_ellipse, minimize_quadratic_on_ellipse
from stpy.helpers.ellipsoid_algorithms import maximize_matrix_quadratic_on_ellipse, minimize_matrix_quadratic_on_ellipse
from stpy.point_processes.poisson import PoissonPointProcess
from stpy.point_processes.poisson_rate_estimator import PositiveRateEstimator
from stpy.borel_set import BorelSet, HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.link_fun_rate_estimator import PermanentalProcessRateEstimator

class LogGaussProcessRateEstimator(PermanentalProcessRateEstimator):


	def __init__(self, *args, **kwargs):
		super().__init__(*args,**kwargs)
		self.discretization = 64

		self.nodes = None
		self.weights = None

	def load_data(self,data):
		super().load_data(data)

		if len(data) > 1:
			weights_arr = []
			nodes_arr = []
			for sample in data:
				(S, obs, dt) = sample
				weights, nodes = S.return_legendre_discretization(self.discretization)
				nodes_arr.append(nodes)
				weights_arr.append(weights*dt)

			self.nodes = self.packing.embed(torch.cat(nodes_arr))
			self.weights = torch.cat(weights_arr)

	def add_data_point(self, new_data):
		super().add_data_point(new_data)

		S, obs, dt = new_data
		weights, nodes = S.return_legendre_discretization(self.discretization)

		if self.nodes is None:
			self.nodes = self.packing.embed(nodes)
			self.weights = weights*dt
		else:
			self.nodes = torch.cat((self.nodes,self.packing.embed(nodes)))
			self.weights = torch.cat((self.weights,weights*dt))


	def sample(self, verbose = False, steps = 10, stepsize = None):

		sigmoid_der_1 = lambda x: torch.exp(-x) / (torch.exp(-x) + 1) ** 2

		if self.data is None:
			self.sampled_theta = torch.zeros(self.get_m()).double().view(-1,1)
			return None


		if self.observations is not None:
			weights = self.weights
			nodes = self.nodes

			nabla = lambda theta: -torch.sum(
				torch.diag(sigmoid_der_1(self.observations @ theta).view(-1) / self.sigmoid(self.observations @ theta).view(-1)) @ self.observations, dim=0).view(-1, 1) \
								  + self.B * torch.sum(torch.diag(weights.view(-1)*sigmoid_der_1(nodes @ theta).view(-1) )@nodes,dim = 0).view(-1,1) +  self.s * theta.view(-1, 1)
		else:
			weights = self.weights
			nodes = self.nodes
			nabla = lambda theta: self.B * torch.sum(torch.diag(weights.view(-1)*sigmoid_der_1(nodes @ theta).view(-1) )@nodes,dim = 0).view(-1,1) +  self.s * theta.view(-1, 1)


		#theta = self.rate.view(-1, 1)*np.nan

	#	while torch.sum(torch.isnan(theta))>0:


		W = self.construct_covariance_matrix_laplace()
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-6))
		eta = 0.5 / (L + 1)
		theta = self.rate.view(-1, 1)
		for k in range(steps):
			s = torch.randn(size=(self.get_m(), 1)).double()
			theta = theta - eta * nabla(theta) + np.sqrt(2 * eta) * s
			if verbose == True:
				print("Iter:", k, theta.T)

		self.sampled_theta = theta
		return None

	def sample_value(self, S):
		"""
		Given a pre-sampled value evaluate certain portions of the domain S
		:param S:
		:return:
		"""
		weights, nodes = S.return_legendre_discretization(64)
		Phi = self.packing.embed(nodes)
		map_vals = torch.sum(weights*self.B*self.sigmoid(Phi @ self.sampled_theta))
		return map_vals

	def sample_path(self, S, n=128):
		xtest = S.return_discretization(n)
		return self.sigmoid(self.packing.embed(xtest) @ self.sampled_theta)*self.B



	def penalized_likelihood(self, threads=4):
		sigmoid = lambda x: 1./(1.+np.exp(-x))
		weights = self.weights.numpy()
		nodes = self.nodes.numpy()
		#times = self.times.numpy()
		#times = self.times.numpy()

		if self.observations is not None:
			observations = self.observations.numpy()
			#loss = lambda theta: float(-np.sum(np.log(self.B * sigmoid(observations @ theta))) \
									   #+ self.B * np.einsum('i,i',(weights ,sigmoid(nodes @ theta))) + self.s * np.sum(theta ** 2))
			loss = lambda theta: float(-np.sum(np.log(self.B * sigmoid(observations @ theta)))\
									   + self.B * np.sum(weights*sigmoid(nodes @ theta).reshape(-1) ) + 0.5*self.s * np.sum(theta ** 2))

		else:
			loss = lambda theta: float(
				+self.B * np.sum(weights * sigmoid(theta @ nodes.T)) + self.s * np.sum(theta ** 2))

		theta = np.zeros(self.get_m())
		res = minimize(loss, theta, jac= None, method='L-BFGS-B',options={'maxcor': 20,'iprint':-1,
																		  'maxfun':150000,'maxls': 50,'ftol':1e-12,
																		  'eps':1e-12,'gtol':1e-8})

		self.rate = torch.from_numpy(res.x)

		return self.rate


	def construct_covariance_matrix_laplace(self):
		sigmoid_der_1 = lambda x: np.exp(-x) / (np.exp(-x) + 1) ** 2
		sigmoid_der_2 = lambda x: 2 * np.exp(-2 * x) / (np.exp(-x) + 1) ** 3 - np.exp(-x) / (np.exp(-x) + 1) ** 2
		sigmoid = lambda x: 1./(1.+np.exp(-x))

		W = torch.zeros(size=(self.get_m(), self.get_m())).double()

		if self.feedback == "count-record":
			if self.observations is not None:
				input = (self.observations@self.rate).view(-1)
				scales = (sigmoid_der_1(input)**2 + sigmoid_der_2(input)*sigmoid(input))/(sigmoid(input)**2)
				W = torch.einsum('ij,i,ik->jk',self.observations,scales, self.observations)

			if self.nodes is not None:
				scales = self.B * sigmoid_der_2(self.nodes@self.rate) * self.weights
				Z = torch.einsum('ij,i,ik->jk', self.nodes, scales, self.nodes)
				W = W + Z

		else:
			raise AssertionError("Not implemented.")
		return W + torch.eye(self.get_m()).double()*self.s


	def mean_var_laplace_set(self, S, dt, beta=2.):
		if self.approx_fit == False:
			self.W = self.construct_covariance_matrix_laplace()
			self.approx_fit = True
			self.W_inv_approx = torch.pinverse(self.W)
		return self.mean_std_per_action(S, self.W, dt, beta)

	def mean_std_per_action(self,S,W, dt , beta):
		weights,nodes = S.return_legendre_discretization(64)
		Phi = self.packing.embed(nodes)
		vars = torch.einsum('ij,jk,ki->i',Phi, self.W_inv_approx, Phi.T)

		vars = (vars + np.abs(vars))/2
		map_vals = weights*self.B*self.sigmoid(Phi @ self.rate)
		lcb_vals = weights*self.B*self.sigmoid(Phi @ self.rate - beta*np.sqrt(vars))
		ucb_vals = weights*self.B*self.sigmoid(Phi @ self.rate + beta*np.sqrt(vars))

		return dt * torch.sum(map_vals), dt * torch.sum(ucb_vals), torch,sum(lcb_vals) * dt


	def map_lcb_ucb(self, S, n, beta = 2.0, delta = 0.01):
		"""
		Calculate exact confidence using laplace approximation on a whole set domain
		:param S: set
		:param n: discretization
		:param beta: beta
		:return:
		"""

		xtest = S.return_discretization(n)
		if self.data is None:
			return self.b+0*xtest[:,0].view(-1,1),self.b+0*xtest[:,0].view(-1,1),self.B+0*xtest[:,0].view(-1,1)

		Phi = self.packing.embed(xtest)
		map = self.B*self.sigmoid(Phi @ self.rate)

		if self.uncertainty == "laplace":
			W = self.construct_covariance_matrix_laplace()
		W_inv = torch.pinverse(W)

		vars = torch.einsum('ij,jk,ki->i',Phi, W_inv, Phi.T)
		lcb = self.B*self.sigmoid(Phi @ self.rate - beta*np.sqrt(vars))
		ucb = self.B*self.sigmoid(Phi @ self.rate + beta*np.sqrt(vars))

		return map, lcb, ucb

	def sigmoid(self, x):
		return 1./(1.+torch.exp(-x))

	def mean_rate(self, S, n=128):
		xtest = S.return_discretization(n)
		return self.sigmoid(self.packing.embed(xtest) @ self.rate)*self.B

