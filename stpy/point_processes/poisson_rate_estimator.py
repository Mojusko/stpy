import cvxpy as cp
import mosek
import numpy as np
import scipy
import torch
from autograd_minimize import minimize
from quadprog import solve_qp
from torchmin import minimize as minimize_torch

from stpy.embeddings.bernstein_embedding import BernsteinEmbedding, BernsteinSplinesEmbedding, \
	BernsteinSplinesOverlapping
from stpy.embeddings.bump_bases import PositiveNystromEmbeddingBump, TriangleEmbedding, FaberSchauderEmbedding
from stpy.embeddings.optimal_positive_basis import OptimalPositiveBasis
from stpy.helpers.ellipsoid_algorithms import maximize_on_elliptical_slice
from stpy.point_processes.rate_estimator import RateEstimator


class PoissonRateEstimator(RateEstimator):

	def __init__(self, process, hierarchy, d=1, m=100, kernel_object=None, B=1., s=1., jitter=10e-8, b=0.,
				 basis='triangle', estimator='likelihood', feedback='count-record', offset=0.1, uncertainty='laplace',
				 approx=None, stepsize=None, embedding=None, beta=2., sampling='proximal+prox', peeking=True,
				 constraints=True, var_cor_on=True,
				 samples_nystrom=15000, inverted_constraint=False, steps=None, dual=True, no_anchor_points=1024, U=1.,
				 opt='torch'):

		self.process = process
		self.d = d
		self.s = s
		self.b = b
		self.B = B
		self.U = U
		self.stepsize = stepsize
		self.sampling = sampling
		self.steps = steps
		self.opt = opt
		self.kernel_object = kernel_object
		# set hierarchy
		self.constraints = constraints
		self.hierarchy = hierarchy
		self.ucb_identified = False
		self.inverted_constraint = inverted_constraint
		# approximation
		self.loglikelihood = 0.
		self.dual = dual
		self.peeking = peeking
		self.no_anchor_points = no_anchor_points
		if beta < 0.:
			self.beta = lambda t: self.beta_theory()
		else:
			self.beta = lambda t: beta
		self.var_cor_on = var_cor_on

		if basis == 'triangle':
			self.packing = TriangleEmbedding(d, m, kernel_object=kernel_object, B=B, b=b, offset=offset,
											 s=np.sqrt(jitter))
		elif basis == 'bernstein':
			self.packing = BernsteinEmbedding(d, m, kernel_object=kernel_object, B=B, b=b, offset=offset,
											  s=np.sqrt(jitter))
		elif basis == 'splines':
			self.packing = BernsteinSplinesEmbedding(d, m, kernel_object=kernel_object, B=B, b=b, offset=offset,
													 s=np.sqrt(jitter))
		elif basis == 'nystrom':
			self.packing = PositiveNystromEmbeddingBump(d, m, kernel_object=kernel_object, B=B, b=b, offset=offset,
														s=np.sqrt(jitter), samples=samples_nystrom)
		elif basis == 'overlap-splines':
			self.packing = BernsteinSplinesOverlapping(d, m, kernel_object=kernel_object, B=B, b=b, offset=offset,
													   s=np.sqrt(jitter))
		elif basis == 'faber':
			self.packing = FaberSchauderEmbedding(d, m, kernel_object=kernel_object, B=B, b=b, offset=offset,
												  s=np.sqrt(jitter))
		elif basis == "optimal-positive":
			self.packing = OptimalPositiveBasis(d, m, kernel_object=kernel_object, B=B, b=b, offset=offset,
												s=np.sqrt(jitter), samples=samples_nystrom)
		elif basis == "custom":
			self.packing = embedding
		else:
			raise NotImplementedError("The request positive basis is not implemented.")
		self.m = m
		self.data = None
		self.covariance = False

		# stabilizing the matrix inversion
		self.jitter = jitter

		# for variance stabilization
		self.stabilization = None
		self.approx_fit = False

		# properties of rate estimator
		self.estimator = estimator
		self.feedback = feedback
		self.uncertainty = uncertainty
		self.approx = approx

		# precompute information
		self.basic_sets = self.hierarchy.get_sets_level(self.hierarchy.levels)

		self.varphis = torch.zeros(size=(len(self.basic_sets), self.get_m())).double()
		self.variances = torch.ones(size=(len(self.basic_sets), 1)).double().view(-1)
		self.variances_histogram = []
		self.observations = None
		self.rate = None
		self.W = (s) * torch.eye(self.get_m()).double()
		self.W_inv_approx = (1. / s) * torch.eye(self.get_m()).double()
		self.beta_value = 2.
		self.sampled_theta = None

		if self.dual == True:
			if self.d == 1:
				anchor = no_anchor_points
				self.anchor_points = self.hierarchy.top_node.return_discretization(anchor)
				self.anchor_weights = torch.zeros(size=(anchor, 1)).double().view(-1)
			elif self.d == 2:
				anchor = no_anchor_points
				self.anchor_points = self.hierarchy.top_node.return_discretization(int(np.sqrt(anchor)))
				self.anchor_weights = torch.zeros(size=(anchor, 1)).double().view(-1)
			self.global_dt = 0.
			self.anchor_points_emb = self.packing.embed(self.anchor_points)

		if feedback == "count-record" and basis != "custom":
			print("Precomputing phis.")
			for index_set, set in enumerate(self.basic_sets):
				self.varphis[index_set, :] = self.packing.integral(set)
				self.variances[index_set] = set.volume() * self.B
		else:
			pass

		print("Precomputation finished.")

	def add_data_point(self, new_data, times=True):

		super().add_data_point(new_data, times=times)

		if self.rate is not None:
			rate = self.rate
		else:
			l, _, u = self.get_constraints()
			Gamma_half = self.cov()
			rate = Gamma_half @ u

		if self.feedback == 'histogram':
			val = self.packing.integral(new_data[0]) @ rate * new_data[2]
			v = - np.log(val) + val

		elif self.feedback == 'count-record':
			v = self.packing.integral(new_data[0]) @ rate * new_data[2]
			if new_data[1] is not None:
				val2 = self.packing.embed(new_data[1]) @ rate * new_data[2]
				v = v - torch.sum(np.log(val2))

		self.loglikelihood += v

	def beta_theory(self):
		if self.approx_fit == False:
			l, Lambda, u = self.get_constraints()
			Gamma_half, invGamma_half = self.cov(inverse=True)

			## norm
			norm = self.s

			## constraints
			eps = 10e-3
			res = Gamma_half @ self.rate.view(-1, 1) - torch.from_numpy(l).view(-1, 1)
			xi = res.clone()
			xi[res > eps] = 0.

			constraint = xi.T @ Gamma_half @ self.W_inv_approx @ Gamma_half.T @ xi

			## concentration
			vol = 4 * np.log(1. / 0.1) + torch.logdet(self.W) - self.get_m() * np.log(self.s)
			self.beta_value = np.sqrt(norm + vol + constraint)
			print('-------------------')
			print("New beta:", self.beta_value)
			print("norm:", norm)
			print("constraint:", constraint)
			print("vol:", vol)
			print("-------------------")
		else:
			pass
		return self.beta_value

	def get_constraints(self):
		return self.packing.get_constraints()

	def cov(self, inverse=False):
		return self.packing.cov(inverse=inverse)

	def fit_gp(self, threads=4):

		if self.data is not None:
			if self.feedback == "count-record":

				if self.estimator == "likelihood":
					if self.opt == 'cvxpy':
						self.penalized_likelihood(threads=threads)
					elif self.opt == 'torch':
						self.penalized_likelihood_fast(threads=threads)
					else:
						raise NotImplementedError("The optimization method does not exist")

				elif self.estimator == "least-sq":
					self.least_squares_weighted()

				elif self.estimator == "bins":
					self.penalized_likelihood_bins()

				else:
					raise AssertionError("wrong name.")


			elif self.feedback == 'histogram':

				if self.estimator == "likelihood":
					self.penalized_likelihood_integral()

				elif self.estimator == "least-sq":
					self.least_squares_weighted_integral()

				elif self.estimator == "bins":
					self.penalized_likelihood_integral_bins()

				else:
					raise AssertionError("wrong name.")
			else:
				raise AssertionError("wrong name.")
		else:
			l, Lambda, u = self.get_constraints()
			Gamma_half = self.cov()
			self.rate = l

	def sample_mirror_langevin(self, steps=500, verbose=False):

		l, Lambda, u = self.get_constraints()
		Gamma_half, invGamma_half = self.cov(inverse=True)

		v = torch.from_numpy((u + l) / 2.).view(-1, 1)
		S = torch.diag(torch.from_numpy(u - l).view(-1) / 2.).double()

		phis = self.phis.clone() @ invGamma_half

		if self.observations is not None:
			obs = self.observations @ invGamma_half
		else:
			obs = None

		invGamma = invGamma_half.T @ invGamma_half
		transform = lambda y: S @ torch.tanh(y) + v

		if self.feedback == "count-record" and self.dual == False:
			if obs is not None:
				func = lambda y: -torch.sum(torch.log(obs @ transform(y)).view(-1)) \
								 + torch.sum(phis @ transform(y)) \
								 + self.s * transform(y).T @ invGamma @ transform(y) + torch.sum(
					torch.log(1. / (1. - transform(y) ** 2)))
			else:
				func = lambda y: torch.sum(phis @ transform(y)) \
								 + self.s * transform(y).T @ invGamma @ transform(y) + torch.sum(
					torch.log(1. / (1. - transform(y) ** 2)))  # torch.sum(torch.log(0.5*(1.+torch.cosh(2*y))))


		elif self.feedback == "count-record" and self.dual == True:
			mask = self.bucketized_counts > 0
			phis = self.varphis[mask, :] @ invGamma_half
			tau = self.total_bucketized_time[mask]

			if obs is not None:
				obs = self.anchor_points_emb @ invGamma_half
				weights = self.anchor_weights
				mask = weights > 0.

				func = lambda y: -torch.sum(weights[mask].view(-1, 1) * torch.log(obs[mask, :] @ transform(y))) \
								 + torch.sum(tau.view(-1, 1) * (phis @ transform(y))) \
								 + self.s * transform(y).T @ invGamma @ transform(y) + torch.sum(
					torch.log(1. / (1. - (transform(y) ** 2))))  # + torch.sum(torch.log(0.5*(1.+torch.cosh(2*y))))
			else:
				func = lambda y: torch.sum(tau.view(-1, 1) * (phis @ transform(y))) \
								 + self.s * transform(y).T @ invGamma @ transform(y) + torch.sum(
					torch.log(1. / (1. - transform(y) ** 2)))  # + torch.sum(torch.log(0.5*(1.+torch.cosh(2*y))))

		elif self.feedback == "histogram":
			func = lambda y: - torch.sum(
				self.counts.clone().view(-1) * torch.log(phis @ (S @ torch.tanh(y) + v)).view(-1)) \
							 + torch.sum(phis @ (S @ torch.tanh(y) + v)) \
							 + self.s * (S @ torch.tanh(y) + v).T @ invGamma @ (S @ torch.tanh(y) + v)

		y = torch.rand(size=(self.get_m(), 1), dtype=torch.float64, requires_grad=True)

		# initiallize with map sqeezed more
		y.data = Gamma_half @ self.rate.view(-1, 1)  # u < theta < l

		u_new = u + 0.01
		l_new = l - 0.01
		v2 = torch.from_numpy((u_new + l_new) / 2.).view(-1, 1)
		S2 = torch.diag(torch.from_numpy(u_new - l_new).view(-1) / 2.).double()
		#
		y.data = torch.inverse(S2) @ (y.data - v2)
		y.data = torch.atanh(y.data)

		W = S.T @ invGamma_half.T @ self.construct_covariance_matrix_laplace() @ invGamma_half @ S
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-8))
		eta = 0.05 / (L + 1)

		print("Eta:", eta)

		for k in range(steps):

			w = torch.randn(size=(self.get_m(), 1)).double()
			nabla_y = torch.autograd.functional.jacobian(func, y).data[0, 0, :, :]
			y.data = y.data - eta * nabla_y + np.sqrt(2 * eta) * w
			theta = torch.tanh(y).detach()

			if verbose == True:
				print("Iter:", k, (S @ theta + v).T)
				print(y.T)

		self.sampled_theta = invGamma_half @ transform(y.data)

	def sample_projected_langevin(self, steps=300, verbose=False, stepsize=None):
		"""
		:param burn_in:
		:return:
		"""

		Gamma_half = self.packing.cov()

		def prox(x):
			z = x.numpy()
			theta = cp.Variable((self.get_m(), 1))
			objective = cp.Minimize(cp.sum_squares(z - theta))
			constraints = []
			l, Lambda, u = self.get_constraints()
			Lambda = Lambda @ Gamma_half.numpy()
			constraints.append(Lambda @ theta >= l.reshape(-1, 1))
			prob = cp.Problem(objective, constraints)
			prob.solve(solver=cp.OSQP, warm_start=False, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
			return torch.from_numpy(theta.value)

		if self.feedback == "count-record" and self.dual == False:
			if self.observations is not None:
				nabla = lambda y: -torch.einsum('i,ij->j', 1. / (self.observations @ y).view(-1),
												self.observations).view(-1, 1) + \
								  torch.sum(self.phis, dim=0).view(-1, 1) \
								  + self.s * y.view(-1, 1)
			else:
				nabla = lambda theta: torch.sum(self.phis, dim=0).view(-1, 1) + self.s * theta.view(-1, 1)

		elif self.feedback == "count-record" and self.dual == True:
			mask = self.bucketized_counts > 0
			phis = self.varphis[mask, :]
			tau = self.total_bucketized_time[mask]

			if self.observations is not None:
				obs = self.anchor_points_emb
				weights = self.anchor_weights
				mask = weights > 0.
				nabla = lambda y: -torch.einsum('i,ij->j', weights[mask] / ((obs[mask, :] @ y).view(-1)),
												obs[mask]).view(-1, 1) + \
								  torch.einsum('i,ij->j', tau, phis).view(-1, 1) \
								  + self.s * y.view(-1, 1)
			else:
				nabla = lambda y: torch.einsum('i,ij->j', tau, phis).view(-1, 1) \
								  + self.s * y.view(-1, 1)


		elif self.feedback == "histogram":
			nabla = lambda theta: -torch.sum(torch.diag((1. / (self.phis @ theta).view(-1)) * self.counts) @ self.phis,
											 dim=0).view(-1, 1) \
								  + torch.sum(self.phis, dim=0).view(-1, 1) + self.s * theta.view(-1, 1)

		theta = self.rate.view(-1, 1)
		W = self.construct_covariance_matrix_laplace(minimal=True)
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-5))

		if stepsize is None:
			eta = 0.5 / (L + 1)
		else:
			eta = np.minimum(1, stepsize * 0.5 / L)

		print(eta)
		for k in range(steps):
			w = torch.randn(size=(self.get_m(), 1)).double()
			theta = prox(theta - eta * nabla(theta) + np.sqrt(2 * eta) * w)

			if verbose == True:
				print("Iter:", k, theta.T)

		self.sampled_theta = theta

	def sample_proximal_langevin_prox(self, steps=300, verbose=False, stepsize=None):
		"""
		:param burn_in:
		:return:
		"""

		Gamma_half, invGamma_half = self.packing.cov(inverse=True)
		# invGamma = invGamma_half.T @ invGamma_half
		l, Lambda, u = self.get_constraints()
		Lambda = Lambda @ Gamma_half.numpy()

		def prox(x):
			res = solve_qp(np.eye(self.get_m()), x.numpy().reshape(-1), C=Gamma_half.numpy(), b=l.numpy(),
						   factorized=True)
			return torch.from_numpy(res[0]).view(-1, 1)

		# theta_n = cp.Variable((self.get_m(), 1))
		# x = cp.Parameter((self.get_m(), 1))
		# objective = cp.Minimize(cp.sum_squares(x - theta_n))
		#
		# constraints = []
		# l, Lambda, u = self.get_constraints()
		# Lambda = Lambda @ Gamma_half.numpy()
		# constraints.append(Lambda @ theta_n >= l.reshape(-1, 1))
		# constraints.append(Lambda @ theta_n <= u.reshape(-1, 1))
		#
		# prob = cp.Problem(objective, constraints)

		# def prox(x):
		# 	return Gamma_half @ torch.from_numpy(scipy.optimize.nnls(invGamma.numpy(), (invGamma_half@x).numpy().reshape(-1), maxiter = 1000)[0]).view(-1,1)

		if self.data is not None:
			if self.feedback == "count-record" and self.dual == False:
				if self.observations is not None:
					nabla = lambda y: -torch.einsum('i,ij->j', 1. / (self.observations @ y).view(-1),
													self.observations).view(-1, 1) + \
									  torch.sum(self.phis, dim=0).view(-1, 1) \
									  + self.s * y.view(-1, 1)
				else:
					nabla = lambda theta: torch.sum(self.phis, dim=0).view(-1, 1) + self.s * theta.view(-1, 1)

			elif self.feedback == "count-record" and self.dual == True:
				mask = self.bucketized_counts > 0
				phis = self.varphis[mask, :]
				tau = self.total_bucketized_time[mask]

				if self.observations is not None:
					obs = self.anchor_points_emb
					weights = self.anchor_weights
					mask = weights > 0.
					nabla = lambda y: -torch.einsum('i,ij->j', weights[mask] / ((obs[mask, :] @ y).view(-1)),
													obs[mask]).view(-1, 1) + \
									  torch.einsum('i,ij->j', tau, phis).view(-1, 1) \
									  + self.s * y.view(-1, 1)
				else:
					nabla = lambda y: torch.einsum('i,ij->j', tau, phis).view(-1, 1) \
									  + self.s * y.view(-1, 1)


			elif self.feedback == "histogram":
				nabla = lambda theta: -torch.sum(
					torch.diag((1. / (self.phis @ theta).view(-1)) * self.counts) @ self.phis,
					dim=0).view(-1, 1) \
									  + torch.sum(self.phis, dim=0).view(-1, 1) + self.s * theta.view(-1, 1)
		else:
			nabla = lambda theta: self.s * theta.view(-1, 1)

		if self.rate is not None:
			theta = self.rate.view(-1, 1)
		else:
			theta = self.b + 0.05 * torch.rand(size=(self.get_m(), 1), dtype=torch.float64, requires_grad=False).view(
				-1, 1) ** 2

		for k in range(steps):
			w = torch.randn(size=(self.get_m(), 1)).double()

			# calculate proper step-size
			W = self.construct_covariance_matrix_laplace(theta=theta)
			L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-3))
			if stepsize is not None:
				eta = 0.5 * stepsize / L
			else:
				eta = 0.5 / L

			# prox calculate
			# x.value = theta.numpy()
			# prob.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-3, eps_rel=1e-3)
			# proximal_theta = torch.from_numpy(theta_n.value)

			# update step
			#			theta = 0.5 * theta - eta * nabla(theta) + 0.5 * proximal_theta + np.sqrt(2 * eta) * w

			# update step
			theta = 0.5 * theta - eta * nabla(theta) + 0.5 * prox(theta) + np.sqrt(2 * eta) * w
			if verbose == True:
				print("Iter:", k, theta.T)

		self.sampled_theta = prox(theta)

	def sample_proximal_langevin_simple_prox(self, steps=300, verbose=False):

		Gamma_half, invGamma_half = self.packing.cov(inverse=True)
		l, Lambda, u = self.get_constraints()
		prox_simple = lambda x: torch.minimum(torch.maximum(x.view(-1), torch.from_numpy(l).view(-1)) \
											  , torch.from_numpy(u).view(-1)).view(-1, 1)

		def prox(x):
			return invGamma_half @ prox_simple(Gamma_half @ x)

		phis = self.phis
		if self.feedback == "count-record" and self.dual == False:
			if self.observations is not None:
				obs = self.observations

				func = lambda y: -torch.sum(torch.log(obs @ y)) \
								 + torch.sum((phis @ y)) \
								 + self.s * y.T @ y

				nabla = lambda y: -torch.einsum('i,ij->j', 1. / (obs @ y).view(-1), obs).view(-1, 1) + \
								  torch.sum(phis, dim=0).view(-1, 1) \
								  + self.s * y.view(-1, 1)
			else:
				func = lambda y: torch.sum(phis @ y).view(-1, 1) \
								 + self.s * y.T @ y

				nabla = lambda y: torch.sum(phis, dim=0).view(-1, 1) + self.s * y.view(-1, 1)





		elif self.feedback == "count-record" and self.dual == True:
			mask = self.bucketized_counts > 0
			phis = self.varphis[mask, :]
			tau = self.total_bucketized_time[mask]

			if self.observations is not None:
				obs = self.anchor_points_emb
				weights = self.anchor_weights
				mask = weights > 0.
				func = lambda y: -torch.sum(weights[mask].view(-1, 1) * torch.log(obs[mask, :] @ y)) \
								 + torch.sum(tau.view(-1, 1) * (phis @ y)) \
								 + self.s * y.T @ y

				nabla = lambda y: -torch.einsum('i,ij->j', weights[mask] / ((obs[mask, :] @ y).view(-1)),
												obs[mask]).view(-1, 1) + \
								  torch.einsum('i,ij->j', tau, phis).view(-1, 1) \
								  + self.s * y.view(-1, 1)
			else:
				func = lambda y: torch.sum(tau.view(-1, 1) * (phis @ y)) \
								 + self.s * y.T @ y

				nabla = lambda y: torch.einsum('i,ij->j', tau, phis).view(-1, 1) \
								  + self.s * y.view(-1, 1)

		elif self.feedback == "histogram":
			func = lambda y: - torch.sum(self.counts.view(-1) * torch.log(phis @ y).view(-1)) + \
							 torch.sum(phis @ y) \
							 + self.s * y.T @ y
			nabla = lambda y: -torch.einsum('i,ij->j', self.counts.view(-1) / (phis @ y).view(-1), phis).view(-1, 1) + \
							  torch.sum(phis, dim=0).view(-1, 1) + self.s * y

		# hessian = lambda y: self.construct_covariance_matrix_laplace()

		y = prox(torch.randn(size=(self.get_m(), 1), dtype=torch.float64, requires_grad=True))
		y.data = self.rate.view(-1, 1)

		W = self.construct_covariance_matrix_laplace()
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-5))

		eta = 0.5 / (L + 1)

		for k in range(steps):
			W = torch.randn(size=(self.get_m(), 1)).double()
			nabla_y = nabla(y.data)
			y.data = (1 - eta) * y.data - eta * nabla_y + eta * prox(y.data) + np.sqrt(2 * eta) * W
			if verbose == True:
				print("Iter:", k, y.T)
				print("grad:", y.grad.T)

		self.sampled_theta = prox(y.detach())

	def sample_hessian_positive_langevin(self, steps=500, verbose=False, stepsize=None):

		if self.data is not None:
			if self.feedback == "count-record" and self.dual == False:
				if self.observations is not None:
					nabla = lambda y: -torch.einsum('i,ij->j', 1. / (self.observations @ y).view(-1),
													self.observations).view(-1, 1) + \
									  torch.sum(self.phis, dim=0).view(-1, 1) \
									  + self.s * y.view(-1, 1)
				else:
					nabla = lambda theta: torch.sum(self.phis, dim=0).view(-1, 1) + self.s * theta.view(-1, 1)

			elif self.feedback == "count-record" and self.dual == True:

				mask = self.bucketized_counts > 0
				phis = self.varphis[mask, :]
				tau = self.total_bucketized_time[mask]

				if self.observations is not None:
					obs = self.anchor_points_emb
					weights = self.anchor_weights
					mask = weights > 0.
					nabla = lambda y: -torch.einsum('i,ij->j', weights[mask] / ((obs[mask, :] @ y).view(-1)),
													obs[mask]).view(-1, 1) + \
									  torch.einsum('i,ij->j', tau, phis).view(-1, 1) \
									  + self.s * y.view(-1, 1)
				else:
					nabla = lambda y: torch.einsum('i,ij->j', tau, phis).view(-1, 1) \
									  + self.s * y.view(-1, 1)


			elif self.feedback == "histogram":
				nabla = lambda theta: -torch.sum(
					torch.diag((1. / (self.phis @ theta).view(-1)) * self.counts) @ self.phis,
					dim=0).view(-1, 1) \
									  + torch.sum(self.phis, dim=0).view(-1, 1) + self.s * theta.view(-1, 1)
		else:
			nabla = lambda theta: self.s * theta.view(-1, 1)

		Gamma_half = self.packing.cov()
		lz, Lambda, u = self.get_constraints()

		Lambda = torch.from_numpy(Lambda) @ Gamma_half
		y = self.b + 0.05 * torch.rand(size=(self.get_m(), 1), dtype=torch.float64, requires_grad=True).view(-1) ** 2

		if self.rate is not None:
			y.data = self.rate.data + Gamma_half @ y.data
		else:
			y.data = Gamma_half @ y.data

		if verbose == True:
			print("initial point")
			print(y.data)

		W = self.construct_covariance_matrix_laplace()
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-5))

		if stepsize is None:
			eta = 1. / (L + 1)
		else:
			eta = stepsize / (L + 1)

		D = lambda x: torch.diag(1. / torch.abs(Lambda @ x).view(-1))
		sqrt_hessian = lambda x: Lambda @ D(x)

		phi = lambda x: -torch.sum(torch.log(Lambda @ x))
		nabla_phi = lambda x: -torch.einsum('i,ij->j', 1. / (Lambda @ x).view(-1), Lambda)
		hessian_phi = lambda x: Lambda.T @ torch.diag(1. / (Lambda @ x).view(-1) ** 2) @ Lambda

		for k in range(steps):
			w = torch.randn(size=(self.get_m(), 1)).double()
			nabla_val = nabla(y)
			H = sqrt_hessian(y.data)
			z = nabla_phi(y.data).view(-1, 1) - eta * nabla_val + np.sqrt(2 * eta) * H @ w

			# y.data = newton_solve(lambda s: nabla_phi(s).reshape(-1)-z.data.reshape(-1),y.reshape(-1),
			#  					  verbose = verbose, grad = hessian_phi).view(-1,1)

			# # minimization appraoch
			def objective(s):
				return torch.sum((nabla_phi(s).reshape(-1) - z.reshape(-1)) ** 2)

			# #

			# x0 = y.reshape(-1).clone().detach().numpy()
			# res = minimize(objective, x0, backend='torch', method='Newton-CG', precision='float64', tol=1e-5, hvp_type='vhp')
			# y.data = torch.from_numpy(res.x)

			x0 = y.reshape(-1).clone()
			res = minimize_torch(objective, x0, method='newton-cg', tol=1e-5)
			y.data = res.x

			if verbose:
				print("Iter:", k)
				print(y.T)

		self.sampled_theta = y.data

	def sample_mla_prime(self, steps=100, verbose=False, stepsize=None):
		Gamma_half, invGamma_half = self.packing.cov(inverse=True)
		invGamma = invGamma_half.T @ invGamma_half
		l, Lambda, u = self.get_constraints()
		Lambda = torch.from_numpy(Lambda) @ Gamma_half

		if self.data is not None:
			if self.feedback == "count-record" and self.dual == False:
				if self.observations is not None:
					observations = self.observations @ invGamma_half
					phis = self.phis @ invGamma_half
					nabla = lambda y: -torch.einsum('i,ij->j', 1. / (observations @ y).view(-1),
													observations).view(-1, 1) + \
									  torch.sum(phis, dim=0).view(-1, 1) \
									  + self.s * invGamma @ y.view(-1, 1)
				else:
					nabla = lambda theta: torch.sum(phis, dim=0).view(-1, 1) + self.s * invGamma @ theta.view(-1, 1)

		else:
			nabla = lambda theta: self.s * invGamma @ theta.view(-1, 1)

		y = self.b + 0.05 * torch.rand(size=(self.get_m(), 1), dtype=torch.float64, requires_grad=True).reshape(-1,
																												1) ** 2
		# if self.rate is not None:
		# 	y.data = Gamma_half @ self.rate.data.view(-1,1) + y.data
		# else:
		y.data = y.data

		if verbose == True:
			print("initial point")
			print(y.data)

		W = invGamma_half.T @ self.construct_covariance_matrix_laplace() @ invGamma_half
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-5))

		if stepsize is None:
			eta = 1. / (L + 1)
		else:
			eta = stepsize / (L + 1)

		from stpy.approx_inference.sampling_helper import get_increment
		for k in range(steps):

			nabla_val = nabla(y)

			# cvxpy minimization
			# x = cp.Variable((self.get_m(), 1))
			# objective = cp.Minimize( eta * nabla_val.detach().numpy().T @ x - cp.sum(cp.log(x)) -(-1./y.data).T@x)
			# constraints = [x >= 0.]
			#
			# prob = cp.Problem(objective, constraints)
			# prob.solve(solver = cp.MOSEK)

			w0 = (eta * nabla_val.data + 1. / y.data)
			# initial point for the solve
			# w0 = -1./( torch.from_numpy(x.value))

			# simulate
			f = lambda w, n: n / torch.abs(w)
			w = get_increment(eta, 1000, f, w0, path=False)

			# back mirror map
			y.data = (-1. / w)

			if verbose:
				print("Iter:", k)
				print(y.T)

		self.sampled_theta = invGamma_half @ y.data

	def sample_hessian_positive_langevin_2(self, steps=500, verbose=False, stepsize=None, preconditioner=True):

		Gamma_half, invGamma_half = self.packing.cov(inverse=True)
		invGamma = invGamma_half @ invGamma_half
		if self.data is not None:

			if self.feedback == "count-record" and self.dual == False:

				observations = self.observations @ invGamma_half
				phis = self.phis @ invGamma_half

				if self.observations is not None:
					nabla = lambda y: -torch.einsum('i,ij->j', 1. / (observations @ y).view(-1),
													observations).view(-1, 1) + \
									  torch.sum(phis, dim=0).view(-1, 1) \
									  + self.s * invGamma @ y.view(-1, 1)
				else:
					nabla = lambda theta: torch.sum(phis, dim=0).view(-1, 1) + self.s * invGamma @ theta.view(-1, 1)

		else:
			nabla = lambda theta: self.s * invGamma @ theta.view(-1, 1)

		y = torch.rand(size=(self.get_m(), 1), dtype=torch.float64, requires_grad=True).view(-1) ** 2
		# if self.rate is not None:
		#	y.data = Gamma_half @ self.rate.data + y.data

		if verbose == True:
			print("initial point")
			print(y.data)

		W = self.construct_covariance_matrix_laplace(minimal=True)
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-5))

		if stepsize is None:
			eta = 1. / (L + 1)
		else:
			eta = stepsize / (L + 1)

		for k in range(steps):
			w = torch.randn(size=(self.get_m(), 1)).double() / torch.abs(y.data).view(-1, 1)
			nabla_val = nabla(y)
			z = -1. / y.data.view(-1, 1) + self.b - eta * Gamma_half @ nabla_val + np.sqrt(2 * eta) * Gamma_half @ w
			y.data = -1. / z + self.b

			if verbose:
				print("Iter:", k)
				print(y.T)

		self.sampled_theta = invGamma_half @ y.data

	def sample_newton_langevin(self, steps=1000, stepsize=None, verbose=False):
		Gamma_half, invGamma_half = self.packing.cov(inverse=True)
		invGamma = invGamma_half @ invGamma_half
		if self.data is not None:

			if self.feedback == "count-record" and self.dual == False:

				observations = self.observations @ invGamma_half
				phis = self.phis @ invGamma_half

				if self.observations is not None:
					nabla = lambda y, bar: -torch.einsum('i,ij->j', 1. / (observations @ y).view(-1),
														 observations).view(-1, 1) + \
										   torch.sum(phis, dim=0).view(-1, 1) \
										   + self.s * invGamma @ y.view(-1, 1) - bar * 1. / y
				else:
					nabla = lambda theta, bar: torch.sum(phis, dim=0).view(-1, 1) + self.s * invGamma @ theta.view(
						-1, 1) - bar * 1. / theta

		else:
			nabla = lambda theta, bar: self.s * invGamma @ theta.view(-1, 1) - bar * 1. / theta

		y = 0.05 * torch.rand(size=(self.get_m(), 1), dtype=torch.float64, requires_grad=True).view(-1, 1) ** 2

		barrier = 10.
		# hessian = lambda theta,bar: torch.einsum('ik,k,kj->ij',observations.T,(observations@theta).view(-1),observations) + invGamma + bar/theta**2
		hessian = lambda theta, bar: observations.T @ torch.diag(
			1 / (observations @ theta).view(-1) ** 2) @ observations + invGamma + torch.diag(bar / theta.view(-1) ** 2)
		hessian_sqrt = lambda theta, bar: torch.cholesky(hessian(theta, bar))
		eta = 1.

		for k in range(steps):
			w = torch.randn(size=(self.get_m(), 1)).double()
			nabla_val = nabla(y, barrier)
			y.data = y.data - torch.linalg.solve(hessian(y.data, barrier), nabla_val) + np.sqrt(
				2 * eta) * torch.linalg.solve(hessian_sqrt(y.data, barrier), w)

			if verbose:
				print("Iter:", k)
				print(y.T)

		self.sampled_theta = invGamma_half @ y.data

	# self.sampled_theta = y.data

	def sample_hmc(self, steps=1000, stepsize=None, verbose=False):
		import hamiltorch
		phis = self.phis
		if self.feedback == "count-record" and self.dual == False:
			if self.observations is not None:
				obs = self.observations
				func = lambda y: torch.sum(torch.log(obs @ y)) \
								 - torch.sum((phis @ y)) \
								 - self.s * y.T @ y
			else:
				func = lambda y: - torch.sum(phis @ y).view(-1, 1) \
								 - self.s * y.T @ y

		num_samples = 1
		num_steps_per_sample = steps
		if stepsize is None:
			step_size = 1e-8
		else:
			step_size = stepsize

		params_init = self.rate
		self.sample_theta = hamiltorch.sample(log_prob_func=func,
											  params_init=params_init,
											  num_samples=num_samples,
											  step_size=step_size,
											  num_steps_per_sample=num_steps_per_sample)
		print(self.sampled_theta)

	def sample_variational(self, xtest, accuracy=1e-4, verbose=False, samples=1):
		from stpy.approx_inference.variational_mf import VMF_SGCP
		cov_params = [self.kernel_object.kappa, self.kernel_object.gamma]
		S_borders = np.array([[-1., 1.]])
		num_inducing_points = self.m
		num_integration_points = 256
		X = self.x

		var_mf_sgcp = VMF_SGCP(S_borders, X, cov_params, num_inducing_points,
							   num_integration_points=num_integration_points,
							   update_hyperparams=False, output=0, conv_crit=accuracy)
		var_mf_sgcp.run()
		sample_paths = var_mf_sgcp.sample_posterior(xtest, num_samples=1.)
		return sample_paths

	def sample(self, verbose=False, steps=1000, domain=None):
		"""
		:return:
		"""
		if self.steps is not None:
			steps = self.steps

		if self.stepsize is not None:
			stepsize = self.stepsize
		else:
			stepsize = None

		l, Lambda, u = self.get_constraints()
		print("Sampling started.")
		if self.rate is None:
			self.fit_gp()

		if self.sampling == 'mirror':
			self.sample_mirror_langevin(steps=steps, verbose=verbose)
		elif self.sampling == 'proximal+prox':
			self.sample_proximal_langevin_prox(steps=steps, verbose=verbose)
		elif self.sampling == "proximal+simple_prox":
			self.sample_proximal_langevin_simple_prox(steps=steps, verbose=verbose)
		elif self.sampling == "hessian":
			self.sample_hessian_positive_langevin(steps=steps, verbose=verbose, stepsize=stepsize)
		elif self.sampling == "hessian2":
			self.sample_hessian_positive_langevin_2(steps=steps, verbose=verbose, stepsize=stepsize)
		elif self.sampling == "mla_prime":
			self.sample_mla_prime(steps=steps, verbose=verbose, stepsize=stepsize)
		elif self.sampling == 'hmc':
			self.sample_hmc(steps=steps, verbose=verbose, stepsize=stepsize)
		elif self.sampling == 'polyia_variational':
			self.sample_variational(accuracy=1. / steps, verbose=verbose)
		else:
			raise NotImplementedError("Sampling of such is not supported.")

		print("Sampling finished.")

	def sampled_lcb_ucb(self, xtest, samples=100, delta=0.1):
		paths = []
		for i in range(samples):
			self.sample()
			path = self.sample_path_points(xtest).view(1, -1)
			paths.append(path)

		paths = torch.cat(paths, dim=0)
		lcb = torch.quantile(paths, delta, dim=0)
		ucb = torch.quantile(paths, 1 - delta, dim=0)
		return lcb, ucb

	def penalized_likelihood_fast(self, threads=4):
		l, Lambda, u = self.get_constraints()
		Gamma_half, invGamma_half = self.cov(inverse=True)

		if self.dual == False:
			# using all points without anchor points
			if self.observations is not None:
				def objective(theta):
					return -torch.sum(torch.log(self.observations @ invGamma_half @ theta)) + torch.sum(
						self.phis @ invGamma_half @ theta) + self.s * 0.5 * torch.sum((invGamma_half @ theta) ** 2)
			else:
				def objective(theta):
					return torch.sum(self.phis @ invGamma_half @ theta) + self.s * 0.5 * torch.sum(
						(invGamma_half @ theta) ** 2)
		else:
			# using anchor points
			mask = self.bucketized_counts > 0
			phis = self.varphis[mask, :]
			tau = self.total_bucketized_time[mask]

			if self.observations is not None:
				observations = self.anchor_points_emb
				weights = self.anchor_weights
				mask = weights > 0.

				def objective(theta):
					return -torch.einsum('i,i', weights[mask],
										 torch.log(observations[mask, :] @ invGamma_half @ theta)) + torch.einsum('i,i',
																												  tau,
																												  phis @ invGamma_half @ theta) + self.s * 0.5 * torch.sum(
						(invGamma_half @ theta) ** 2)
			else:
				def objective(theta):
					return torch.einsum('i,i', tau, phis @ invGamma_half @ theta) + self.s * 0.5 * torch.sum(
						(invGamma_half @ theta) ** 2)

		if self.rate is not None:
			theta0 = torch.zeros(size=(self.get_m(), 1)).view(-1).double()
			theta0.data = self.rate.data
		else:
			theta0 = torch.zeros(size=(self.get_m(), 1)).view(-1).double()

		eps = 1e-4
		res = minimize(objective, theta0.numpy(), backend='torch', method='L-BFGS-B',
					   bounds=(l[0] + eps, u[0]), precision='float64', tol=1e-8,
					   options={'ftol': 1e-08,
								'gtol': 1e-08, 'eps': 1e-08,
								'maxfun': 15000, 'maxiter': 15000,
								'maxls': 20})

		self.rate = invGamma_half @ torch.from_numpy(res.x)
		print(res.message)
		return self.rate

	def penalized_likelihood(self, threads=4):

		theta = cp.Variable(self.get_m())
		l, Lambda, u = self.get_constraints()

		Gamma_half = self.cov(inverse=False)

		if self.dual == False:

			# using all points without anchor points
			phis = self.phis.numpy()
			if self.observations is not None:
				observations = self.observations.numpy()
				objective = cp.Minimize(-cp.sum(cp.log(observations @ theta)) +
										cp.sum(phis @ theta) + self.s * 0.5 * cp.sum_squares(theta))
			else:
				objective = cp.Minimize(cp.sum(phis @ theta) + self.s * 0.5 * cp.sum_squares(theta))

		else:

			# using anchor points
			mask = self.bucketized_counts.clone().numpy() > 0
			phis = self.varphis[mask, :].clone().numpy()
			tau = self.total_bucketized_time[mask].clone().numpy()

			if self.observations is not None:
				observations = self.anchor_points_emb.numpy()
				weights = self.anchor_weights.numpy()
				mask = weights > 0.
				objective = cp.Minimize(-cp.sum(cp.multiply(weights[mask], cp.log(observations[mask, :] @ theta))) +
										cp.sum(cp.multiply(tau, phis @ theta)) + self.s * 0.5 * cp.sum_squares(theta))
			else:
				objective = cp.Minimize(cp.sum(cp.multiply(tau, phis @ theta)) + self.s * 0.5 * cp.sum_squares(theta))

		constraints = []

		Lambda = Lambda @ Gamma_half.numpy()

		constraints.append(Lambda @ theta >= l)
		constraints.append(Lambda @ theta <= u)

		prob = cp.Problem(objective, constraints)

		if self.rate is not None:
			theta.value = self.rate.numpy()

		try:
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
					   mosek_params={mosek.iparam.num_threads: threads,
									 mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-4,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-4,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-4})

			self.rate = torch.from_numpy(theta.value)
			return self.rate
		except:
			print("Optimization failed. Using the old value.")
			print(prob.status)
			return self.rate

	def penalized_likelihood_integral(self, threads=4):

		phis = self.phis.numpy()
		counts = self.counts.numpy()

		theta = cp.Variable(self.get_m())
		l, Lambda, u = self.get_constraints()
		Gamma_half = self.cov().numpy()
		objective = cp.Minimize(-cp.sum(counts @ cp.log(phis @ theta)) + cp.sum(phis @ theta)
								+ self.s * 0.5 * cp.sum_squares(theta))

		constraints = []
		Lambda = Lambda @ Gamma_half
		constraints.append(Lambda @ theta >= l)
		constraints.append(Lambda @ theta <= u)

		# if self.rate is not None:
		#	theta.value = self.rate.numpy()
		try:
			prob = cp.Problem(objective, constraints)
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
					   mosek_params={mosek.iparam.num_threads: threads,
									 mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-4,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-4,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-4})
			self.rate = torch.from_numpy(theta.value)
		except:
			print("Optimization failed. Using the old value.")
			print(prob.status)

		return self.rate

	def bucketization(self):

		phis = []
		observations = []

		# project sets to smallest forms, and then sum on those only
		basic_sets = self.basic_sets

		data_basic = [[] for _ in range(len(basic_sets))]
		sensing_times = [[] for _ in range(len(basic_sets))]
		counts = torch.zeros(len(basic_sets)).int()
		total_data = 0.
		self.total_bucketized_obs = torch.zeros(size=(len(basic_sets), 1)).double().view(-1)
		self.total_bucketized_time = torch.zeros(size=(len(basic_sets), 1)).double().view(-1)

		for sample in self.data:
			S, obs, dt = sample
			if obs is not None:
				total_data = total_data + obs.size()[0]  # total counts
				for index, elementary in enumerate(basic_sets):  # iterate over basic sets
					mask = elementary.is_inside(obs)  # mask which belong to the elementary
					if S.inside(elementary) == True:
						data_basic[index].append(obs[mask])
						counts[index] += 1
						sensing_times[index].append(dt)
			else:
				for index, elementary in enumerate(basic_sets):
					if S.inside(elementary) == True:
						data_basic[index].append(torch.Tensor([]))
						counts[index] += 1
						sensing_times[index].append(dt)

		for index, elementary in enumerate(basic_sets):
			arr = np.array([int(elem.size()[0]) for elem in data_basic[index]])  # counts over sensing rounds
			phi = self.packing.integral(elementary)  # * counts[index]

			self.total_bucketized_obs[index] = float(np.sum(arr))
			self.total_bucketized_time[index] = float(np.sum(sensing_times[index]))

			observations.append(arr)
			phis.append(phi.view(1, -1))  # construct varphi_B

		self.bucketized_obs = observations.copy()  # these are number of counts associated with sensings
		self.bucketized_time = sensing_times.copy()  # these are times each basic set has been sensed
		self.bucketized_counts = counts  # these are count each basic set has been sensed

	def variance_correction(self, variance):

		if self.var_cor_on == 1:

			g = lambda B, k, mu: -0.5 * (B ** 2) / ((mu ** 2) * k) - B / (mu * k) + (np.exp(B / (k * mu)) - 1)
			gn = lambda k: g(self.U, k, variance)

			from scipy import optimize
			k = optimize.bisect(gn, 1, 10000000)

			return k
		else:
			return 1.

	def least_squares_weighted(self, threads=4):

		# if self.approx_fit == False:
		# 	self.bucketization()

		theta = cp.Variable(self.get_m())
		l, Lambda, u = self.get_constraints()
		Gamma_half = self.cov().numpy()

		mask = self.bucketized_counts.clone().numpy() > 0
		observations = self.total_bucketized_obs[mask].clone().numpy()
		phis = self.varphis[mask, :].clone().numpy()
		tau = self.total_bucketized_time.clone().numpy()

		variances = self.variances.view(-1).clone().numpy()

		for i in range(variances.shape[0]):
			if mask[i] > 0:
				variances[i] = variances[i] * tau[i] * self.variance_correction(variances[i] * tau[i])

		selected_variances = variances[mask]
		objective = cp.Minimize(
			cp.sum_squares((cp.multiply((phis @ theta), tau[mask]) - observations) / (np.sqrt(selected_variances)))
			+ 0.5 * self.s * cp.norm2(theta) ** 2)

		constraints = []
		Lambda = Lambda @ Gamma_half
		# constraints.append(Lambda @ theta >= l)
		constraints.append(Lambda @ theta <= u)

		prob = cp.Problem(objective, constraints)

		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
				   mosek_params={mosek.iparam.num_threads: threads,
								 mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
								 mosek.dparam.intpnt_co_tol_pfeas: 1e-4,
								 mosek.dparam.intpnt_co_tol_dfeas: 1e-4,
								 mosek.dparam.intpnt_co_tol_rel_gap: 1e-4})
		print(prob.status)
		self.rate = torch.from_numpy(theta.value)
		return self.rate

	def least_sqaures_weighted_fast(self, threads=4):

		l, Lambda, u = self.get_constraints()
		Gamma_half, invGamma_half = self.cov(inverse=True)

		mask = self.bucketized_counts > 0
		observations = self.total_bucketized_obs[mask]
		phis = self.varphis[mask, :]
		tau = self.total_bucketized_time

		variances = self.variances.view(-1)
		for i in range(variances.size()[0]):
			if mask[i] > 0:
				variances[i] = variances[i] * tau[i] * self.variance_correction(variances[i] * tau[i])
		selected_variances = variances[mask]

		def objective(theta):
			return torch.sum(
				((tau[mask] * (phis @ invGamma_half @ theta) - observations) / (np.sqrt(selected_variances))) ** 2) \
				   + self.s * 0.5 * torch.sum((invGamma_half @ theta) ** 2)

		if self.rate is not None:
			theta0 = torch.zeros(size=(self.get_m(), 1)).view(-1).double()
			theta0.data = Gamma_half @ self.rate.data
		else:
			theta0 = torch.zeros(size=(self.get_m(), 1)).view(-1).double()

		eps = 1e-4
		res = minimize(objective, theta0.numpy(), backend='torch', method='L-BFGS-B',
					   bounds=(l[0] + eps, u[0]), precision='float64', tol=1e-8,
					   options={'ftol': 1e-06,
								'gtol': 1e-06, 'eps': 1e-08,
								'maxfun': 15000, 'maxiter': 15000,
								'maxls': 20})
		self.rate = invGamma_half @ torch.from_numpy(res.x)

		return self.rate

	def least_squares_weighted_integral(self, threads=4):

		# if self.approx_fit == False:
		# 	self.bucketization()

		theta = cp.Variable(self.get_m())
		l, Lambda, u = self.get_constraints()
		Gamma_half = self.cov().numpy()

		phis = self.phis.clone().numpy()  # integrated actions
		if self.rate is None:
			rate = torch.pinverse(torch.from_numpy(Gamma_half)) @ torch.from_numpy(u)
		else:
			rate = self.rate.clone()

		if len(self.variances_histogram) > 0:
			variances = self.variances_histogram.numpy()

			for i in range(variances.shape[0]):
				variances[i] = variances[i] * self.variance_correction(variances[i])
		else:
			variances = np.zeros(len(self.data))
			i = 0
			for S, obs, dt in self.data:
				variances[i] = S.volume() * self.B
				variances[i] = variances[i] * self.variance_correction(variances[i])
				i = i + 1

		observations = self.counts.clone().numpy()

		objective = cp.Minimize(cp.sum_squares((phis @ theta - observations) / np.sqrt(variances))
								+ self.s * cp.sum_squares(theta))
		constraints = []
		Lambda = Lambda @ Gamma_half
		constraints.append(Lambda @ theta >= l)
		constraints.append(Lambda @ theta <= u)
		prob = cp.Problem(objective, constraints)

		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
				   mosek_params={mosek.iparam.num_threads: threads,
								 mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
								 mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
								 mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
								 mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})

		self.rate = torch.from_numpy(theta.value)

		return self.rate

	def penalized_likelihood_bins(self, threads=4):
		theta = cp.Variable(self.get_m())
		l, Lambda, u = self.get_constraints()
		Gamma_half = self.cov().numpy()

		mask = self.bucketized_counts.clone().numpy() > 0
		observations = self.total_bucketized_obs[mask].clone().numpy()
		phis = self.varphis[mask, :].clone().numpy()
		tau = self.total_bucketized_time[mask].clone().numpy()

		constraints = []
		Lambda = Lambda @ Gamma_half
		constraints.append(Lambda @ theta >= l)
		constraints.append(Lambda @ theta <= u)

		objective = cp.Minimize(
			-cp.sum(observations @ cp.log(cp.multiply(tau, phis @ theta))) + cp.sum(cp.multiply(phis @ theta, tau))
			+ self.s * 0.5 * cp.sum_squares(theta))
		prob = cp.Problem(objective, constraints)
		try:
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
					   mosek_params={mosek.iparam.num_threads: threads,
									 mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-8})

			self.rate = torch.from_numpy(theta.value)
		except:
			print("optimization failed.")
		return self.rate

	def penalized_likelihood_integral_bins(self, threads=4):
		phis = self.phis.numpy()
		counts = self.counts.numpy()

		theta = cp.Variable(self.get_m())
		l, Lambda, u = self.get_constraints()
		Gamma_half = self.cov().numpy()
		objective = cp.Minimize(-cp.sum(counts @ cp.log(phis @ theta)) + cp.sum(phis @ theta)
								+ self.s * 0.5 * cp.sum_squares(theta))

		constraints = []
		Lambda = Lambda @ Gamma_half
		constraints.append(Lambda @ theta >= l)
		constraints.append(Lambda @ theta <= u)

		try:
			if constraints:
				prob = cp.Problem(objective, constraints)
			else:
				prob = cp.Problem(objective)
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
					   mosek_params={mosek.iparam.num_threads: threads,
									 mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})
			self.rate = torch.from_numpy(theta.value)
		except:
			print("Optimization failed. Using the old value.")

		return self.rate

	def update_variances(self, value=False, force=False):
		self.approx_fit = True
		if (self.feedback == "count-record" and self.estimator=="least-sq") or force == True:
			print("updating variance")
			for index, set in enumerate(self.basic_sets):
				if value == False:
					ucb = self.ucb(set)
					self.variances[index] = np.minimum(ucb, self.variances[index])
				else:
					self.variances[index] = self.mean_set(set)
		else:
			if self.data is not None:
				if self.peeking == True:
					new_var = []
					for S, _, dt in self.data:
						new_var.append(float(self.ucb(S)) * dt)
					self.variances_histogram = torch.Tensor(new_var.copy()).double()
				else:
					last = self.data[-1]
					new_var = torch.Tensor([self.ucb(last[0]) * last[2]]).double()
					if len(self.variances_histogram) > 0:
						self.variances_histogram = torch.cat((self.variances_histogram, new_var))
					else:
						self.variances_histogram = new_var
		self.approx_fit = False

	def ucb(self, S, dt=1., delta=0.5):

		if self.data is None or self.rate is None:
			return self.B * S.volume() * dt

		if self.approx == None:

			if self.uncertainty == "laplace":
				return self.mean_var_laplace_set(S, dt=dt, beta=self.beta(0))[1]

			elif self.uncertainty == "least-sq":
				return self.mean_var_reg_set(S, dt=dt, beta=self.beta(0))[1]

			elif self.uncertainty == "bins":
				return self.mean_var_bins_set(S, dt=dt, beta=self.beta(0))[1]

			elif self.uncertainty == "likelihood-ratio":
				return self.mean_var_ratio_set(S, dt=dt, beta=self.beta(0))[1]

			elif self.uncertainty == "conformal":
				return self.mean_var_conformal_set(S, dt=dt, delta=delta)[2]

			else:
				raise AssertionError("Not Implemented.")

		elif self.approx == "ellipsoid":

			if self.approx_fit == False:
				self.fit_ellipsoid_approx()
				self.beta(0)
				print("Fitting Approximation.")
				self.approx_fit = True
			return self.map_lcb_ucb_approx_action(S, dt=dt, beta=self.beta(0))[2]
		else:
			raise AssertionError("Not implemented.")

	def mean_std_per_action(self, S, W, dt, beta):

		phi = self.packing.integral(S) * dt
		Gamma_half = self.cov().numpy()

		l, Lambda, u = self.get_constraints()

		Lambda = Lambda @ Gamma_half
		ucb, _ = maximize_on_elliptical_slice(phi.numpy(), (W).numpy(), self.rate.view(-1).numpy(), beta, l, Lambda, u)
		lcb, _ = maximize_on_elliptical_slice(-phi.numpy(), (W).numpy(), self.rate.view(-1).numpy(), beta, l, Lambda, u)
		map = phi @ self.rate

		return map, float(ucb), -float(lcb)

	def mean_var_laplace_set(self, S, dt, beta=2.):
		if self.approx_fit == False:
			self.W = self.construct_covariance_matrix_laplace()
			self.approx_fit = True
		return self.mean_std_per_action(S, self.W, dt, beta)

	def mean_var_reg_set(self, S, dt, beta=2.):
		if self.approx_fit == False:
			self.W = self.construct_covariance_matrix_regression()
			self.approx_fit = True
		return self.mean_std_per_action(S, self.W, dt, beta)

	def mean_var_bins_set(self, S, dt, beta=2.):
		if self.approx_fit == False:
			self.W = self.construct_covariance_matrix_bins()
			self.approx_fit = True
		return self.mean_std_per_action(S, self.W, dt, beta)

	def mean_var_ratio_set(self, S, dt, beta=2.):
		x = self.packing.integral(S) * dt
		map = x @ self.rate
		# v = np.log(1. / 0.1) - torch.sum(self.counts.double() @ torch.log(self.phis.double() @ self.rate)) \
		#	+ torch.sum(self.phis.double() @ self.rate) + 0.5 * self.s * torch.norm(self.rate) ** 2
		v = np.log(1. / 0.1) + self.likelihood + 0.5 * self.s * torch.norm(self.rate) ** 2

		phis = self.phis.numpy()
		counts = self.counts.numpy()
		theta = cp.Variable(self.get_m())
		l, Lambda, u = self.get_constraints()
		Gamma_half = self.cov().numpy()

		objective_min = cp.Minimize(x @ theta)
		objective_max = cp.Maximize(x @ theta)

		constraints = []
		Lambda = Lambda @ Gamma_half
		constraints.append(Lambda @ theta >= l)
		constraints.append(Lambda @ theta <= u)

		constraints.append(
			-cp.sum(counts @ cp.log(phis @ theta)) + cp.sum(phis @ theta) + self.s * 0.5 * cp.sum_squares(
				theta) <= v)

		prob = cp.Problem(objective_min, constraints)
		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False)
		lcb = np.dot(theta.value, x)
		prob = cp.Problem(objective_max, constraints)
		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False)
		ucb = np.dot(theta.value, x)

		return map, ucb, lcb

	def map_lcb_ucb_approx_action(self, S, dt=1., beta=2.):
		phi = self.packing.integral(S)
		map = dt * phi @ self.rate

		ucb = map + beta * np.sqrt(phi @ self.W_inv_approx @ phi.T)
		# ucb = np.minimum(dt * ucb, self.B * S.volume() * dt)

		lcb = map - beta * np.sqrt(phi @ self.W_inv_approx @ phi.T)
		# lcb = np.maximum(dt * lcb, self.b * S.volume() * dt)
		return map, lcb, ucb

	def fit_ellipsoid_approx(self):

		if self.uncertainty == "laplace":
			self.W = self.construct_covariance_matrix_laplace()
		elif self.uncertainty == 'least-sq':
			self.W = self.construct_covariance_matrix_regression()
		elif self.uncertainty == 'bins':
			self.W = self.construct_covariance_matrix_bins()
		else:
			raise AssertionError("Not implemented.")

		self.W_inv_approx = torch.pinverse(self.W)

	def construct_covariance_matrix(self):
		if self.estimator == "likelihood":
			self.W = self.construct_covariance_matrix_laplace()
		elif self.estimator == "least-sq":
			self.W = self.construct_covariance_matrix_regression()
		elif self.estimator == "bins":
			self.W = self.construct_covariance_matrix_bins()
		else:
			raise NotImplementedError("This estimator is not implemented.")
		return self.W

	def construct_covariance_matrix_laplace(self, theta=None):
		W = torch.zeros(size=(self.get_m(), self.get_m())).double()

		if self.feedback == "count-record":

			if self.observations is not None:

				if theta is None:
					D = torch.diag(1. / ((self.observations @ self.rate).view(-1) ** 2))
					W = self.observations.T @ D @ self.observations
				else:
					D = torch.diag(1. / ((self.observations @ theta).view(-1) ** 2))
					W = self.observations.T @ D @ self.observations

		elif self.feedback == "histogram":
			# D = torch.diag(self.counts / (self.phis @ self.rate).view(-1) ** 2)
			if len(self.variances_histogram) > 0:
				variances = self.variances_histogram.view(-1).clone()

				for i in range(variances.shape[0]):
					variances[i] = variances[i] * self.variance_correction(variances[i])

				D = torch.diag(self.counts / variances ** 2)

			W = self.phis.T @ D @ self.phis
		else:
			raise AssertionError("Not implemented.")

		return W + torch.eye(self.get_m()).double() * self.s

	def construct_covariance_matrix_regression(self):
		W = torch.zeros(size=(self.get_m(), self.get_m())).double()
		if self.data is not None:
			variances = self.variances
			if self.feedback == "count-record":
				mask = self.bucketized_counts > 0
				tau = self.total_bucketized_time
				for index_o, o in enumerate(self.bucketized_obs):
					n = mask[index_o]
					if n > 0:
						A = self.varphis[index_o, :].view(-1, 1) @ self.varphis[index_o, :].view(1, -1) * tau[index_o]
						k = self.variance_correction(tau[index_o] * variances[index_o])
						W = W + A / (variances[index_o] * k)

			elif self.feedback == "histogram":

				if len(self.variances_histogram) > 0:
					variances = self.variances_histogram.view(-1).clone()

					for i in range(variances.shape[0]):
						variances[i] = variances[i] * self.variance_correction(variances[i])

					D = torch.diag(1. / variances)
					W = self.phis.T @ D @ self.phis

		return W + torch.eye(self.get_m()).double() * self.s

	def construct_covariance_matrix_bins(self):
		W = torch.zeros(size=(self.get_m(), self.get_m())).double()

		if self.feedback == "count-record":

			mask = self.bucketized_counts > 0
			tau = self.total_bucketized_time
			varphis = self.varphis[mask, :]
			variances = self.variances.view(-1).clone()

			for i in range(variances.size()[0]):
				if mask[i] > 0:
					variances[i] = variances[i] * self.variance_correction(variances[i] * tau[i])

			variances = variances[mask]
			tau = tau[mask]

			if self.observations is not None:
				D = torch.diag(tau / variances)
				W = varphis.T @ D @ varphis

		elif self.feedback == "histogram":

			if len(self.variances_histogram) > 0:
				variances = self.variances_histogram.view(-1).clone()

				for i in range(variances.shape[0]):
					variances[i] = variances[i] * self.variance_correction(variances[i])

				D = torch.diag(1. / variances)
				W = self.phis.T @ D @ self.phis
		else:
			raise AssertionError("Not implemented.")

		return W + torch.eye(self.get_m()).double() * self.s

	def gap(self, S, actions, w, dt, beta=2.):
		"""
		Estimates the gap of an action S,
		:param S:
		:param dt:
		:return:
		"""
		phi = self.packing.integral(S) * dt
		Gamma_half = self.packing.cov().numpy()

		if self.approx is None:
			l, Lambda, u = self.get_constraints()
			Lambda = Lambda @ Gamma_half
			ucbs = []
			for action in actions:
				phi_a = self.packing.integral(action) * dt
				# ucb, _ = maximize_on_elliptical_slice(phi_a.numpy()-phi.numpy(), self.W.numpy(), self.rate.view(-1).numpy(), beta, l, Lambda, u)
				ucb, _ = maximize_on_elliptical_slice(phi.numpy(), self.W.numpy(),
													  self.rate.view(-1).numpy(), beta, l, Lambda, u)
				ucbs.append(float(ucb))
			gap = torch.max(torch.Tensor(ucbs))

		else:
			if self.data is None:
				return (self.B - self.b) * S.volume()

			if self.ucb_identified == False:
				print("Recomputing UCB.....")
				self.ucb_identified = True
				self.fit_ellipsoid_approx()
				self.max_ucb = -1000
				self.ucb_action = None

				for action in actions:
					_, __, ucb = self.map_lcb_ucb_approx_action(action, dt=dt, beta=self.beta(0))
					ucb = ucb / w(action)

					if ucb > self.max_ucb:
						self.max_ucb = ucb
						self.ucb_action = action

			map, lcb, ucb = self.map_lcb_ucb_approx_action(S, dt=dt, beta=self.beta(0))
			gap = w(S) * self.max_ucb - lcb
		return gap

	def information(self, S, dt, precomputed=None):

		if self.data is None:
			return 1.

		if self.W is None:
			self.construct_covariance_matrix()

		if self.feedback == "count-record":
			varphi_UCB = self.packing.integral(self.ucb_action).view(1, -1) * dt

			if precomputed is not None:
				Upsilon = precomputed[S] * dt
			else:
				ind = []
				for index, set in enumerate(self.basic_sets):
					if S.inside(set):
						ind.append(index)
				Upsilon = self.varphis[ind, :] * dt

			I = torch.eye(Upsilon.size()[0]).double()
			G = self.W_inv_approx - self.W_inv_approx @ Upsilon.T @ torch.inverse(
				I + Upsilon @ Upsilon.T) @ Upsilon @ self.W_inv_approx
			return 10e-4 + torch.logdet(varphi_UCB @ self.W_inv_approx @ varphi_UCB.T) - torch.logdet(
				varphi_UCB @ G @ varphi_UCB.T)

		elif self.feedback == "histogram":

			return torch.log(1 + self.packing.integral(S) @ self.W_inv_approx @ self.packing.integral(S) * dt ** 2)

	def map_lcb_ucb_approx(self, S, n, beta=2.0, delta=0.01):
		xtest = S.return_discretization(n)
		if self.data is None:
			return self.b + 0 * xtest[:, 0].view(-1, 1), \
				   self.b + 0 * xtest[:, 0].view(-1, 1), \
				   self.B + 0 * xtest[:, 0].view(-1, 1)

		self.fit_ellipsoid_approx()
		self.fit_ellipsoid_approx()

		Phi = self.packing.embed(xtest).double()
		map = Phi @ self.rate
		N = Phi.size()[0]

		ucb = torch.zeros(size=(N, 1)).double()
		lcb = torch.zeros(size=(N, 1)).double()

		for i in range(N):
			x = Phi[i, :].view(-1, 1)
			ucb[i, 0] = np.minimum(map[i] + beta * np.sqrt(x.T @ self.W_inv_approx @ x), self.B)
			lcb[i, 0] = np.maximum(map[i] - beta * np.sqrt(x.T @ self.W_inv_approx @ x), self.b)
		return map, lcb, ucb

	def map_lcb_ucb(self, S, n, beta=2.0):
		"""
		Calculate exact confidence using laplace approximation on a whole set domain
		:param S: set
		:param n: discretization
		:param beta: beta
		:return:
		"""

		xtest = S.return_discretization(n)
		if self.data is None:
			return self.b + 0 * xtest[:, 0].view(-1, 1), \
				   self.b + 0 * xtest[:, 0].view(-1, 1), \
				   self.B + 0 * xtest[:, 0].view(-1, 1)

		N = xtest.size()[0]
		Phi = self.packing.embed(xtest)
		map = Phi @ self.rate

		if self.uncertainty == "laplace":
			W = self.construct_covariance_matrix_laplace()
		elif self.uncertainty == "least-sq":
			W = self.construct_covariance_matrix_regression()
		elif self.uncertainty == "bins":
			W = self.construct_covariance_matrix_bins()
		else:
			raise AssertionError("Not implemented ")

		Gamma_half = self.cov().numpy()
		l, Lambda, u = self.get_constraints()
		Lambda = Lambda @ Gamma_half
		ucb = torch.zeros(size=(N, 1)).double()
		lcb = torch.zeros(size=(N, 1)).double()

		for i in range(N):
			x = Phi[i, :]
			ucbi, _ = maximize_on_elliptical_slice(x.numpy(), (W).numpy(), self.rate.view(-1).numpy(), np.sqrt(beta), l,
												   Lambda,
												   u)
			lcbi, _ = maximize_on_elliptical_slice(-x.numpy(), (W).numpy(), self.rate.view(-1).numpy(), np.sqrt(beta),
												   l, Lambda,
												   u)
			ucb[i, 0] = ucbi
			lcb[i, 0] = -lcbi

		return map, lcb, ucb

	def map_lcb_ucb_likelihood_ratio(self, S, n, delta=0.1, current=False):
		xtest = S.return_discretization(n)

		if self.data is None:
			return self.b + 0 * xtest[:, 0].view(-1, 1), \
				   self.b + 0 * xtest[:, 0].view(-1, 1), \
				   self.B + 0 * xtest[:, 0].view(-1, 1)

		N = xtest.size()[0]
		Phi = self.packing.embed(xtest)
		map = Phi @ self.rate

		ucb = torch.zeros(size=(N, 1)).double()
		lcb = torch.zeros(size=(N, 1)).double()

		phis = self.phis.numpy()

		if current:
			if self.observations is not None:
				v = np.log(1. / delta) - torch.sum(torch.log(self.observations @ self.rate)) + torch.sum(
					self.phis @ self.rate) + self.s * 0.5 * torch.sum(self.rate ** 2)
			else:
				v = np.log(1. / delta) + torch.sum(
					self.phis @ self.rate) + self.s * 0.5 * torch.sum(self.rate ** 2)
		else:
			if self.feedback == 'count-record':
				v = np.log(1. / delta) + self.loglikelihood + 0.5 * self.s * torch.sum(self.rate ** 2)
			elif self.feedback == 'histogram':
				v = np.log(1. / delta) + self.loglikelihood + 0.5 * self.s * torch.sum(self.rate ** 2)
			else:
				raise NotImplementedError("Not compatible with given feedback model ")

		l, Lambda, u = self.get_constraints()
		Gamma_half = self.cov().numpy()
		Lambda = Lambda @ Gamma_half

		for i in range(N):
			x = Phi[i, :].numpy()

			theta = cp.Variable(self.get_m())

			objective_min = cp.Minimize(x @ theta)
			objective_max = cp.Maximize(x @ theta)

			constraints = []
			constraints.append(Lambda @ theta >= l)
			constraints.append(Lambda @ theta <= u)

			if self.feedback == 'count-record':
				if self.observations is not None:
					observations = self.observations.numpy()

					constraints.append(
						-cp.sum(cp.log(observations @ theta)) +
						cp.sum(phis @ theta) + self.s * 0.5 * cp.sum_squares(theta)
						<= v)
				else:
					constraints.append(cp.sum(phis @ theta) + self.s * 0.5 * cp.sum_squares(theta)
									   <= v)

			elif self.feedback == 'histogram':
				constraints.append(
					-cp.sum(cp.log(phis @ theta)) +
					cp.sum(phis @ theta) + self.s * 0.5 * cp.sum_squares(theta)
					<= v)
			else:
				raise NotImplementedError("Does not exist.")

			prob = cp.Problem(objective_min, constraints)
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False)
			lcb[i, 0] = float(np.dot(theta.value, x))

			prob = cp.Problem(objective_max, constraints)
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False)
			ucb[i, 0] = float(np.dot(theta.value, x))

		return map, lcb, ucb

	def mean_var_conformal_set(self, S, dt, beta=2., max_val=None, delta=0.05):
		# self.bucketize_prepare()
		if max_val is None:
			max_val = int(self.B * self.basic_sets[0].volume() * dt) + 1
		map, lcb, ucb = self.conformal_confidence_set(S, delta=delta, max_val=max_val, dt=dt)
		return map, lcb, ucb

	def conformal_score_func(self, theta, new, index):

		if new[1] is None:
			n_new = 0
		else:
			n_new = new[1].size()[0]

		varphi = self.packing.integral(new[0]) * new[2]
		err_new = abs(float(n_new) - float(varphi @ theta))
		n = len(self.bucketized_obs[index])

		if n > 0:

			phis = self.varphis[index].repeat(n, 1)
			res = torch.Tensor(self.bucketized_obs[index]).double()

			err = torch.abs(res - (phis @ theta.view(-1, 1)).view(-1))

			return torch.sum(err < err_new).double() / float(n + 1.) + 1. / (float(n) + 1.)

		else:
			return 0.

	def conformal_confidence(self, delta=0.05, max_val=20, dt=1, step=1):
		lcb = []
		ucb = []
		map = []

		if self.data is not None:
			self.bucketization(time=True)

		for S in self.basic_sets:
			m, u, l = self.conformal_confidence_set(S, delta=delta, max_val=max_val, dt=dt, step=step)

			map.append(m)
			ucb.append(u)
			lcb.append(l)

		return torch.Tensor(map).double(), torch.Tensor(ucb).double(), torch.Tensor(lcb).double()

	def conformal_confidence_set(self, S, delta=0.05, max_val=20, dt=1., step=1):
		"""
		:return: (lcb,ucb)
		"""

		if self.data is not None:
			if self.feedback == "count-record":
				self.penalized_likelihood()
			elif self.feedback == "histogram":
				self.penalized_likelihood_integral()

			# identify the set in basic sets
			index = 0
			for set in self.basic_sets:
				if set.inside(S):
					break
				index += 1

			# calculate map estimate
			map = float(self.rate @ self.packing.integral(S))
		else:
			map = self.b
			return map, self.B, self.b

		scores = []
		j = 0
		score = 1.
		lowest = 0
		n = float(len(self.bucketized_obs[index]))

		while score > np.ceil((1 - delta) * (n + 1)) / (n + 1) and j <= max_val:
			lowest = j
			if j > 0:
				obs = torch.zeros(size=(j, self.d)).double()
				for i in range(self.d):
					obs[:, i] = torch.from_numpy(np.random.uniform(S.bounds[i, 0], S.bounds[i, 1], size=j))
			else:
				obs = None

			# new observation
			new = (S, obs, dt)

			old_phis, old_observations, old_counts = self.add_data_point_and_remove(new)

			if self.feedback == "count-record":
				theta_new = self.penalized_likelihood()
			elif self.feedback == "histogram":
				theta_new = self.penalized_likelihood_integral()

			# restore back the data
			self.phis = old_phis
			self.observations = old_observations
			self.counts = old_counts

			# calculate the score
			score = self.conformal_score_func(theta_new, new, index)
			n = float(len(self.bucketized_obs[index]))

			print(j, "/", max_val, score, np.ceil((1 - delta) * (n + 1)) / (n + 1))
			j = j + 1

		j = max_val
		score = 1.
		largest = max_val

		while score > np.ceil((1 - delta) * (n + 1)) / (n + 1) and j > lowest:
			largest = j
			if j > 0:
				obs = torch.zeros(size=(j, self.d)).double()
				for i in range(self.d):
					obs[:, i] = torch.from_numpy(np.random.uniform(S.bounds[i, 0], S.bounds[i, 1], size=j))
			else:
				obs = None

			# new observation
			new = (S, obs, dt)

			old_phis, old_observations, old_counts = self.add_data_point_and_remove(new)

			if self.feedback == "count-record":
				theta_new = self.penalized_likelihood()
			elif self.feedback == "histogram":
				theta_new = self.penalized_likelihood_integral()

			# restore back the data
			self.phis = old_phis
			self.observations = old_observations
			self.counts = old_counts

			# calculate the score
			score = self.conformal_score_func(theta_new, new, index)
			n = float(len(self.bucketized_obs[index]))

			print(j, "/", max_val, score, np.ceil((1 - delta) * (n + 1)) / (n + 1))
			j = j - 1
		# scores = np.array(scores)
		# mask = scores < np.ceil((1-delta)*(n+1))/(n+1)

		# if np.sum(mask) == 0:
		# 	lowest = 0
		# 	largest = max_val
		# else:
		# 	lowest = np.min(np.arange(0,max_val,step)[mask])
		# 	largest = np.max(np.arange(0, max_val, step)[mask])

		lcb = lowest / dt / S.volume()
		ucb = largest / dt / S.volume()

		return (map, ucb, lcb)
