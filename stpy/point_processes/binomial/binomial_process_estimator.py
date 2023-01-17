import cvxpy as cp
import mosek
import numpy as np
import scipy
import torch
from quadprog import solve_qp

from stpy.borel_set import HierarchicalBorelSets
from stpy.embeddings.bump_bases import TriangleEmbedding
from stpy.kernels import KernelFunction
from stpy.point_processes.rate_estimator import RateEstimator


class BernoulliRateEstimator(RateEstimator):
	"""
		without link function, but with inequality constraints
	"""

	def __init__(self, hierarchy, d=1, m=100, kernel_object=None, B=1., s=1., jitter=10e-8, b=0., basis='triangle',
				 offset=0.1, uncertainty='laplace'):

		self.d = d
		self.s = s
		self.b = b
		self.B = B
		self.uncertainty = uncertainty
		self.hierarchy = hierarchy
		self.kernel_object = kernel_object
		self.packing = TriangleEmbedding(d, m, kernel_object=kernel_object, B=1., b=0., offset=offset,
										 s=np.sqrt(jitter))
		self.feedback = "histogram"
		self.data = None

		self.basic_sets = self.hierarchy.get_sets_level(self.hierarchy.levels)
		self.varphis = torch.zeros(size=(len(self.basic_sets), self.get_m())).double()

		for index_set, set in enumerate(self.basic_sets):
			self.varphis[index_set, :] = self.embed_set(set)

	def embed_set(self, S):
		return self.packing.integral(S).view(1, -1)

	def load_data(self, data):
		"""

		:param data: (S, no_events, out_of, duration, time)
		:return:
		"""
		self.data = []
		self.phis = None
		for datapoint in data:
			self.add_data_point(datapoint)

	def add_data_point(self, datapoint):

		if self.data is None:
			self.load_data([datapoint])
		else:

			# add
			self.data.append(datapoint)

			S, count, pool, duration, time = datapoint
			phi = self.embed_set(S)

			if self.phis is not None:
				self.counts = torch.cat((self.counts, torch.Tensor([count])))
				self.pool = torch.cat((self.pool, torch.Tensor([pool])))
				self.phis = torch.cat((self.phis, phi), dim=0)
			else:
				self.counts = torch.Tensor([count]).double()
				self.pool = torch.Tensor([pool]).double()
				self.phis = phi

	def nabla(self, theta):
		# defining objective
		if self.data is not None:
			return - torch.einsum('i,ij,i->j', self.counts, self.phis, 1. / (self.phis @ theta).view(-1)).view(-1, 1) + \
				   torch.einsum('i,ij,i->j', self.pool - self.counts, self.phis,
								1. / (1. - self.phis @ theta).view(-1)).view(-1, 1) \
				   + self.s * theta.view(-1, 1)
		else:
			return self.s * theta.view(-1, 1)

	def sample(self, steps=10, verbose=False):
		"""
		Langevin dynamics to sample from constrained GP prior

		:param steps: Number of iterations
		:return:
		"""
		l = np.zeros(shape=(len(self.basic_sets)))
		u = np.zeros(shape=(len(self.basic_sets))) + 1.

		# prox operator
		def prox(x):
			res = solve_qp(np.eye(self.get_m()), x.numpy().reshape(-1),
						   C=np.vstack((-self.varphis.numpy(), self.varphis.numpy())).T,
						   b=np.hstack((-u, l)), factorized=True)
			return torch.from_numpy(res[0]).view(-1, 1)

		# initialization
		if self.rate is not None:
			theta = self.rate.view(-1, 1)
		else:
			theta = self.b + 0.05 * torch.rand(size=(self.get_m(), 1), dtype=torch.float64, requires_grad=False).view(
				-1, 1) ** 2

		# loop
		for k in range(steps):
			w = torch.randn(size=(self.get_m(), 1)).double()

			# calculate proper step-size
			W = self.construct_covariance(theta=theta)
			L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-3))
			eta = 0.5 / L

			theta = 0.5 * theta - eta * self.nabla(theta) + 0.5 * prox(theta) + np.sqrt(2 * eta) * w
			if verbose == True:
				print("Iter:", k, theta.T)

		self.sampled_theta = prox(theta)

	def construct_covariance(self, theta):
		D1 = torch.diag(self.counts / (self.phis @ theta).view(-1) ** 2)
		D2 = torch.diag((self.pool - self.counts) / (1 - self.phis @ theta).view(-1) ** 2)

		W = self.phis.T @ (D1 + D2) @ self.phis + self.s * torch.eye(self.get_m()).double()
		return W

	def construct_confidence(self):
		self.W = self.construct_covariance(self.rate)
		self.invW = torch.pinverse(self.W)

	def construct_likelihood_ratio(self, method='full'):
		# for data
		phis = self.phis.numpy()
		counts = self.counts.numpy()

		# for constraints
		varphis = self.varphis.numpy()

		# current fit
		mean_theta = self.rate.numpy()

		if method == 'split':
			pass
		elif method == 'full':
			self.likelihood = - counts @ np.log(phis @ mean_theta) - (1 - counts) @ np.log(1 - phis @ mean_theta) \
							  + self.s * 0.5 * np.sum(mean_theta - 0.5) ** 2
		elif method == 'cv':
			pass

	def ucb(self, S, beta=8., delta=0.1):
		if self.uncertainty == 'laplace':
			ucb = self.embed_set(S) @ self.rate + beta * self.embed_set(S) @ self.invW @ self.embed_set(S).T
			return torch.minimum(torch.Tensor([[1.]]).double(), ucb)

		elif self.uncertainty == "ratio":
			phi = self.embed_set(S)
			phis = self.phis.numpy()
			varphis = self.varphis.numpy()

			counts = self.counts.numpy()
			theta = cp.Variable(self.get_m())

			objective = cp.Maximize(phi @ theta)

			v = np.log(1. / delta) + self.likelihood
			constraints = [- counts @ cp.log(phis @ theta) - (1 - counts) @ cp.log(1 - phis @ theta)
						   + self.s * 0.5 * cp.sum_squares(theta - 0.5) <= v]

			# every set has probability between 0-1.
			constraints.append(varphis @ theta >= np.zeros(varphis.shape[0]))
			constraints.append(varphis @ theta <= np.ones(varphis.shape[0]))

			prob = cp.Problem(objective, constraints)
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
					   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})
			return torch.minimum(torch.Tensor([[1.]]).double(), torch.from_numpy(np.array(prob.value)))

	def lcb(self, S, beta=8., delta=0.1):
		if self.uncertainty == 'laplace':
			lcb = self.embed_set(S) @ self.rate - beta * self.embed_set(S) @ self.invW @ self.embed_set(S).T
			return torch.maximum(torch.Tensor([[0.]]).double(), lcb)

		elif self.uncertainty == "ratio":
			phi = self.embed_set(S)
			phis = self.phis.numpy()
			varphis = self.varphis.numpy()

			counts = self.counts.numpy()
			theta = cp.Variable(self.get_m())

			objective = cp.Minimize(phi @ theta)
			v = np.log(1. / delta) + self.likelihood
			constraints = [- counts @ cp.log(phis @ theta) - (1 - counts) @ cp.log(1 - phis @ theta)
						   + self.s * 0.5 * cp.sum_squares(theta - 0.5) <= v]

			# every set has probability between 0-1.
			constraints.append(varphis @ theta >= np.zeros(varphis.shape[0]))
			constraints.append(varphis @ theta <= np.ones(varphis.shape[0]))

			prob = cp.Problem(objective, constraints)
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
					   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})

			return torch.maximum(torch.Tensor([[0.]]).double(), torch.from_numpy(np.array(prob.value)))

	def fit_gp(self, threads=4):

		phis = self.phis.numpy()
		varphis = self.varphis.numpy()

		counts = self.counts.numpy()
		theta = cp.Variable(self.get_m())
		objective = cp.Minimize(- counts @ cp.log(phis @ theta) - (1 - counts) @ cp.log(1 - phis @ theta)
								+ self.s * 0.5 * cp.sum_squares(theta - 0.5))

		# probability constraints
		constraints = []

		# every set has probability between 0-1.
		constraints.append(varphis @ theta >= np.zeros(varphis.shape[0]))
		constraints.append(varphis @ theta <= np.ones(varphis.shape[0]))

		prob = cp.Problem(objective, constraints)
		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
				   mosek_params={mosek.iparam.num_threads: threads,
								 mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
								 mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
								 mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
								 mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})
		self.rate = torch.from_numpy(theta.value)
		return self.rate


class LinkBernoulliRateEstimator(BernoulliRateEstimator):

	def construct_covariance(self, theta):
		D1 = torch.diag(self.counts / (self.phis @ theta).view(-1) ** 2)
		D2 = torch.diag((self.pool - self.counts) / (1 - self.phis @ theta).view(-1) ** 2)

		W = self.phis.T @ (D1 + D2) @ self.phis + self.s * torch.eye(self.get_m()).double()
		return W

	def log_marginal(self, kernel, X):
		func = kernel.get_kernel()
		K = func(self.x, self.x, **X) + torch.eye(self.n, dtype=torch.float64) * self.s * self.s

		L = torch.linalg.cholesky(K)
		logdet = -0.5 * 2 * torch.sum(torch.log(torch.diag(L)))
		alpha = torch.solve(self.y, K)[0]
		logprob = -0.5 * torch.mm(torch.t(self.y), alpha) + logdet
		logprob = -logprob
		return logprob

	def construct_likelihood_ratio(self, method='full'):
		# for data
		phis = self.phis.numpy()
		counts = self.counts.numpy()

		# for constraints
		varphis = self.varphis.numpy()

		# current fit
		mean_theta = self.rate.numpy()

		if method == 'split':
			pass
		elif method == 'full':
			self.likelihood = - counts @ phis @ mean_theta + np.log(1 + np.exp(phis @ mean_theta)) \
							  + self.s * 0.5 * np.sum(mean_theta) ** 2
		elif method == 'cv':
			pass

	def fit_gp(self, threads=4):
		phis = self.phis.numpy()

		counts = self.counts.numpy()
		theta = cp.Variable(self.get_m())
		objective = cp.Minimize(-cp.sum(cp.multiply(counts, phis @ theta)) + cp.sum(cp.logistic(phis @ theta))
								+ self.s * 0.5 * cp.sum_squares(theta))

		# probability constraints
		constraints = []

		prob = cp.Problem(objective, constraints)
		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
				   mosek_params={mosek.iparam.num_threads: threads,
								 mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
								 mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
								 mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
								 mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})
		self.rate = torch.from_numpy(theta.value)
		return self.rate

	def link(self, x):
		return 1. / (1. + torch.exp(-x))

	def mean_set(self, S):
		return self.link(self.embed_set(S) @ self.rate)

	def ucb(self, S, beta=8., delta=0.1):
		if self.uncertainty == "laplace":
			ucb = self.embed_set(S) @ self.rate + beta * self.embed_set(S) @ self.invW @ self.embed_set(S).T
			return self.link(ucb)
		elif self.uncertainty == "martingale":
			phi = self.embed_set(S)
			hat_theta = self.rate.numpy()

			def constraint_value_gradient(theta, beta):
				y = cp.Variable(self.get_m())
				v = (theta - hat_theta)
				objective2 = cp.Maximize(y @ v - cp.sum(cp.abs(self.phis @ y)) - beta)

				prob = cp.Problem(objective2)
				prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
						   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
										 mosek.dparam.intpnt_co_tol_pfeas: 1e-4,
										 mosek.dparam.intpnt_co_tol_dfeas: 1e-4,
										 mosek.dparam.intpnt_co_tol_rel_gap: 1e-4})
				print(prob.status)
				return prob.value, y.value

			beta = 2.
			iters = 10
			gamma = 0.00000001
			theta = hat_theta
			print(theta)

			for k in range(iters):
				print("Iter:", k)
				d = cp.Variable(self.get_m())
				objective = cp.Minimize(phi @ d.T)
				val, nabla = constraint_value_gradient(theta, beta)
				constraints = [val + nabla.reshape(1, -1) @ d <= 0., cp.sum_squares(d) <= gamma]
				prob = cp.Problem(objective, constraints)
				prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False)
				theta = theta + d.value
				print(theta)

			return phi @ theta

		elif self.uncertainty == "ratio":
			phi = self.embed_set(S)
			phis = self.phis.numpy()

			counts = self.counts.numpy()
			theta = cp.Variable(self.get_m())

			objective = cp.Maximize(phi @ theta)
			v = np.log(1. / delta) + self.likelihood
			constraints = [-cp.sum(cp.multiply(counts, phis @ theta)) + cp.sum(cp.logistic(phis @ theta))
						   + self.s * 0.5 * cp.sum_squares(theta) <= v]

			prob = cp.Problem(objective, constraints)
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
					   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})
			return self.link(phi @ theta.value)

	def lcb(self, S, beta=8., delta=0.1):
		if self.uncertainty == "laplace":
			lcb = self.embed_set(S) @ self.rate - beta * self.embed_set(S) @ self.invW @ self.embed_set(S).T
			return self.link(lcb)
		elif self.uncertainty == "ratio":
			phi = self.embed_set(S)
			phis = self.phis.numpy()

			counts = self.counts.numpy()
			theta = cp.Variable(self.get_m())

			objective = cp.Minimize(phi @ theta)
			v = np.log(1. / delta) + self.likelihood
			constraints = [-cp.sum(cp.multiply(counts, phis @ theta)) + cp.sum(cp.logistic(phis @ theta))
						   + self.s * 0.5 * cp.sum_squares(theta) <= v]

			prob = cp.Problem(objective, constraints)
			prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
					   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
									 mosek.dparam.intpnt_co_tol_pfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_dfeas: 1e-6,
									 mosek.dparam.intpnt_co_tol_rel_gap: 1e-6})
			return self.link(phi @ theta.value)

	def nabla(self, theta):
		if self.data is not None:
			return -torch.einsum('i,ij->j', self.counts, self.phis).view(-1, 1) + \
				   torch.einsum('i,ij,i->j', self.pool, self.phis,
								1. / (1. + torch.exp(self.phis @ theta).view(-1))).view(-1, 1) \
				   + self.s * theta.view(-1, 1)
		else:
			return self.s * theta.view(-1, 1)

	def construct_covariance(self, theta):
		W = torch.eye(self.get_m()).double() * self.s + torch.einsum('i,ij,ik->jk',
																	 torch.exp(self.phis @ theta).view(-1) / (
																				 1 + torch.exp(self.phis @ theta)).view(
																		 -1) ** 2, self.phis, self.phis)
		return W


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	from stpy.point_processes.binomial.binomial_process import BernoulliPointProcess

	d = 1
	gamma = 0.1
	n = 64
	m = 128
	levels = 7
	k = KernelFunction(gamma=gamma, kappa=1.)

	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	actions = hierarchical_structure.get_sets_level(levels)
	dummy = torch.zeros(size=(1, d)).double()

	estimator = BernoulliRateEstimator(hierarchical_structure, m=64, kernel_object=k, s=0.001, uncertainty='ratio')
	estimator_link = LinkBernoulliRateEstimator(hierarchical_structure, m=64, kernel_object=k, s=0.001,
												uncertainty="ratio")

	rate = lambda S: np.sin(np.pi * S.return_discretization(n=1) ** 2) * 0.5
	process = BernoulliPointProcess(hierarchical_structure.get_sets_level(levels), rate=rate)

	N = 100

	data = []
	for i in range(N):
		data.append(process.sample(actions[torch.randint(0, len(actions), size=(1, 1))]))

	estimator.load_data(data)
	estimator_link.load_data(data)

	estimator.fit_gp()
	estimator_link.fit_gp()

	# plot observations
	for datapoint in data:
		S, v, _, _, _ = datapoint
		x = S.return_discretization(n)
		if v == 1:
			plt.plot(x, x * 0, 'ko')
		else:
			plt.plot(x, x * 0, 'ro')

	xtest = hierarchical_structure.top_node.return_discretization(64)
	plt.plot(xtest, estimator.mean_rate(hierarchical_structure.top_node, 64) * actions[0].volume(), 'tab:blue')

	samples = 0
	for i in range(samples):
		estimator.sample(steps=100, verbose=False)
		plt.plot(xtest, estimator.sample_path(hierarchical_structure.top_node, 64) * actions[0].volume(), 'g--')

	estimator.construct_confidence()
	estimator.construct_likelihood_ratio()

	estimator_link.construct_confidence()
	estimator_link.construct_likelihood_ratio()
	# plot function
	for action in actions:
		val = estimator.mean_set(action)
		val_link = estimator_link.mean_set(action)

		ucb, lcb = float(estimator.ucb(action)), float(estimator.lcb(action))
		ucb_link, lcb_link = float(estimator_link.ucb(action, delta=0.5)), float(estimator_link.lcb(action, delta=0.5))
		x = action.return_discretization(64)
		plt.plot(x, x * 0 + rate(action), color='tab:red')
		x = x.view(-1)

		plt.plot(x, x * 0 + val, color='tab:blue', linestyle='--')
		plt.plot(x, x * 0 + val_link, color='tab:pink', linestyle='--')
		plt.fill_between(x, x * 0 + lcb, x * 0 + ucb, color='tab:blue', alpha=0.2)
		plt.fill_between(x, x * 0 + lcb_link, x * 0 + ucb_link, color='tab:pink', alpha=0.2)

	plt.show()
