import cvxpy as cp
import matplotlib.pyplot as plt
import mosek
import numpy as np
import torch

from stpy.borel_set import BorelSet, HierarchicalBorelSets
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson import PoissonPointProcess
from stpy.point_processes.poisson.link_fun_rate_estimator import PermanentalProcessRateEstimator


class MBRPositiveEstimator(PermanentalProcessRateEstimator):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		if self.feedback == "count-record":
			self.varLambdas_vec = torch.zeros(
				size=(self.varLambdas.size()[0], self.varLambdas.size()[1] * self.varLambdas.size()[2])).double()
			for i in range(self.varLambdas.size()[0]):
				self.varLambdas_vec[i, :] = self.varLambdas[i, :, :].reshape(-1)

		self.approx_solver = True

	def fit_gp(self, threads=4):
		if self.data is not None:
			super().fit_gp(threads=threads)
		else:
			self.rate = None

	def mean_rate(self, S, n=128):
		xtest = S.return_discretization(n)
		emb = self.packing.embed(xtest)
		mu = torch.einsum('ij,jk,ik->i', emb, self.rate, emb).view(-1, 1)
		return mu

	def rate_value(self, x, dt=1):
		emb = self.packing.embed(x) * dt
		mu = torch.einsum('ij,jk,ik->i', emb, self.rate, emb).view(-1, 1)
		return mu

	def mean_set(self, S, dt=1.):
		if self.data is not None:
			emb = self.product_integral(S) * dt
			mu = torch.trace(emb @ self.rate).view(1, 1)
		else:
			mu = self.b * S.volume()
		return mu

	def penalized_likelihood(self, threads=4):
		sumLambda = self.sumLambda.numpy()
		Theta = cp.Variable((self.get_m(), self.get_m()), symmetric=True)

		if self.observations is not None:
			observations = self.observations.numpy()
			# cost = cp.sum_squares(cp.diag(emb @ A @ emb.T) - y.view(-1).numpy()) / (self.s ** 2) + (self.lam) * cp.norm(A, "fro")
			objective = -cp.sum(cp.log(observations @ Theta @ observations.T)) + \
						cp.trace(sumLambda @ Theta) + self.s * cp.sum_squares(cp.vec(Theta))
		else:
			objective = cp.trace(sumLambda @ Theta) + self.s * cp.sum_squares(cp.vec(Theta))

		# if self.get_m() == 2:
		# 	# use Lorentz-cone special result
		# 	constraints = [cp.SOC(Theta[0,0]+Theta[1,1],Theta[1,1]    )]
		# else:
		# 	constraints = [Theta >> 0]
		constraints = []
		prob = cp.Problem(cp.Minimize(objective), constraints)

		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
				   mosek_params={mosek.iparam.num_threads: threads,
								 mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
								 mosek.dparam.intpnt_co_tol_pfeas: 1e-3,
								 mosek.dparam.intpnt_co_tol_dfeas: 1e-3,
								 mosek.dparam.intpnt_co_tol_rel_gap: 1e-3})
		self.rate = torch.from_numpy(Theta.value)
		return self.rate

	def penalized_likelihood_bins(self, threads=4):
		Theta = cp.Variable((self.get_m(), self.get_m()), symmetric=True)

		mask = self.bucketized_counts.clone().numpy() > 0
		observations = self.total_bucketized_obs[mask].clone().numpy()
		tau = self.total_bucketized_time[mask].clone().numpy()
		varLambdas_vec = self.varLambdas_vec[mask, :].clone().numpy()

		objective = -cp.sum(observations @ cp.log(cp.multiply(tau, varLambdas_vec @ cp.vec(Theta)))) + \
					cp.sum(cp.multiply(tau, varLambdas_vec @ cp.vec(Theta))) + self.s * cp.sum_squares(cp.vec(Theta))

		constraints = [Theta >> 0]
		prob = cp.Problem(cp.Minimize(objective), constraints)

		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
				   mosek_params={mosek.iparam.num_threads: threads,
								 mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
								 mosek.dparam.intpnt_co_tol_pfeas: 1e-3,
								 mosek.dparam.intpnt_co_tol_dfeas: 1e-3,
								 mosek.dparam.intpnt_co_tol_rel_gap: 1e-3})
		self.rate = torch.from_numpy(Theta.value)
		return self.rate

	def least_squares_weighted(self, threads=4):

		if self.approx_fit == False:
			self.bucketization()

		Theta = cp.Variable((self.get_m(), self.get_m()), symmetric=True)

		mask = self.bucketized_counts.clone().numpy() > 0
		observations = self.total_bucketized_obs[mask].clone().numpy()
		tau = self.total_bucketized_time.clone().numpy()

		# varsumLambdas
		varLambdas_vec = self.varLambdas_vec[mask, :].clone().numpy()

		variances = self.variances.view(-1).clone().numpy()

		for i in range(variances.shape[0]):
			if mask[i] > 0:
				variances[i] = variances[i] * tau[i] * self.variance_correction(variances[i] * tau[i])

		selected_variances = variances[mask]

		objective = cp.sum_squares((varLambdas_vec @ cp.vec(Theta) +
									- observations) / np.sqrt(selected_variances)) + self.s * cp.sum_squares(
			cp.vec(Theta)) / 2
		constraints = [Theta >> 0]
		prob = cp.Problem(cp.Minimize(objective), constraints)

		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
				   mosek_params={mosek.iparam.num_threads: threads,
								 mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
								 mosek.dparam.intpnt_co_tol_pfeas: 1e-3,
								 mosek.dparam.intpnt_co_tol_dfeas: 1e-3,
								 mosek.dparam.intpnt_co_tol_rel_gap: 1e-3})

		self.rate = torch.from_numpy(Theta.value)
		return self.rate

	def construct_covariance_matrix(self):
		if self.estimator == "bins":
			self.construct_covariance_matrix_bins()
		elif self.estimator == "least-sq":
			self.construct_covariance_matrix_regression()
		else:
			raise NotImplementedError("Covariance not implemented")

	def construct_covariance_matrix_regression(self):
		varLambdas = self.varLambdas_vec.clone()
		variances = self.variances
		mask = self.bucketized_counts > 0
		tau = self.total_bucketized_time
		W = torch.zeros(size=(self.get_m() ** 2, self.get_m() ** 2)).double()
		I = torch.eye(self.get_m() ** 2).double()
		W_inv = self.s * torch.eye(self.get_m() ** 2).double()

		for index_o, o in enumerate(self.bucketized_obs):
			n = mask[index_o]
			if n > 0:
				k = self.variance_correction(tau[index_o] * variances[index_o])
				v = tau[index_o] / (variances[index_o] * k)

				vec = varLambdas[index_o, :].view(-1, 1)
				A = vec @ vec.T
				W = W + A * v
				denom = 1. + v * vec.T @ W_inv @ vec
				W_inv = W_inv @ (I - v * vec @ (vec.T @ W_inv) / denom)

		self.W = W + self.s * torch.eye(self.get_m() ** 2).double()
		self.W_inv = W_inv
		# self.W_cholesky = torch.cholesky(self.W, upper=True)
		return self.W

	def construct_covariance_matrix_bins(self):
		self.construct_covariance_matrix_regression()

	def mean_var_reg_set(self, S, dt=1., beta=2., lcb_compute=False):

		if self.data is None:
			return S.volume() * self.b, S.volume() * self.B, S.volume() * self.b

		if self.approx_fit == False:
			self.W = self.construct_covariance_matrix()
			self.approx_fit = True

		map = None
		lcb = None

		if self.approx_solver == True:
			ucb = self.band_no_opt(S, beta=beta, dt=dt, maximization=True)
			if lcb_compute == True:
				lcb = self.band_no_opt(S, beta=beta, dt=dt, maximization=False)
		else:
			ucb = self.band(S, beta=beta, dt=dt, maximization=True)
			if lcb_compute == True:
				lcb = self.band(S, beta=beta, dt=dt, maximization=False)

		return map, ucb, lcb

	def mean_var_bins_set(self, S, dt=1., beta=2., lcb_compute=False):
		return self.mean_var_reg_set(S, dt=dt, beta=beta, lcb_compute=lcb_compute)

	def band(self, S, beta=2., dt=1., maximization=True):
		emb = self.product_integral(S) * dt
		A = cp.Variable((self.get_m(), self.get_m()), symmetric=True)
		cost = cp.trace(A @ emb)
		Z = self.W_cholesky.clone()
		zero = np.zeros(self.get_m() ** 2)
		constraints = [cp.SOC(zero.T @ cp.vec(A) + self.s * beta ** 2, Z @ (cp.vec(A) - cp.vec(self.rate.numpy())))]
		constraints += [A >> 0]

		if maximization == True:
			prob = cp.Problem(cp.Maximize(cost), constraints)
		else:
			prob = cp.Problem(cp.Minimize(cost), constraints)

		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=False,
				   mosek_params={mosek.iparam.num_threads: 4,
								 mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
								 mosek.dparam.intpnt_co_tol_pfeas: 1e-3,
								 mosek.dparam.intpnt_co_tol_dfeas: 1e-3,
								 mosek.dparam.intpnt_co_tol_rel_gap: 1e-3})
		ucb = torch.trace(torch.from_numpy(A.value) @ emb)
		return ucb

	def band_no_opt(self, S, beta=2., dt=1., maximization=True):

		if self.rate is None:
			if maximization == True:
				return S.volume() * dt * self.B
			else:
				return S.volume() * dt * self.b
		else:
			emb = self.product_integral(S)
			cost = torch.trace(self.rate @ emb)
			if maximization == True:
				out = cost + beta * emb.view(1, -1) @ self.W_inv @ emb.view(-1, 1)
			else:
				out = np.maximum(cost - beta * emb.view(1, -1) @ self.W_inv @ emb.view(-1, 1), 0.)
			return out * dt

	def gap(self, S, actions, w, dt, beta=2.):
		"""
		Estimates the gap of an action S,
		:param S:
		:param dt:
		:return:
		"""

		if self.data is None:
			return (self.B - self.b) * S.volume() / w(S)

		if self.ucb_identified == False:
			print("Recomputing UCB.....")
			self.ucb_identified = True
			self.max_ucb = -1000
			self.ucb_action = None
			for action in actions:
				_, ucb, __ = self.mean_var_reg_set(action, dt=dt, beta=self.beta(0))
				ucb = ucb / w(action)
				if ucb > self.max_ucb:
					self.max_ucb = ucb
				self.ucb_action = action
		map, ucb, lcb = self.mean_var_reg_set(S, dt=dt, beta=self.beta(0), lcb_compute=True)
		gap = w(S) * self.max_ucb - lcb
		return gap

	def information(self, S, dt, precomputed=None):

		if self.data is None:
			return 1.

		if self.W is None:
			self.construct_covariance_matrix()

		if self.feedback == "count-record":
			varphi_UCB = self.product_integral(self.ucb_action).view(1, -1) * dt

			ind = []
			for index, set in enumerate(self.basic_sets):
				if S.inside(set):
					ind.append(index)
			Upsilon = self.varLambdas_vec[ind, :] * dt

			I = torch.eye(Upsilon.size()[0]).double()
			G = self.W_inv - self.W_inv @ Upsilon.T @ torch.inverse(I + Upsilon @ Upsilon.T) @ Upsilon @ self.W_inv
			return 10e-4 + torch.logdet(varphi_UCB @ self.W_inv @ varphi_UCB.T) - torch.logdet(
				varphi_UCB @ G @ varphi_UCB.T)

		elif self.feedback == "histogram":
			raise NotImplementedError("Not implemented.")


if __name__ == "__main__":
	torch.manual_seed(2)
	np.random.seed(2)
	d = 1
	gamma = 0.2
	n = 64
	B = 4.
	b = 0.5

	process = PoissonPointProcess(d=1, B=B, b=b)
	Sets = []
	levels = 3
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	Sets = hierarchical_structure.get_all_sets()

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	m = 32
	embedding = HermiteEmbedding(m=m, d=1, gamma=gamma)
	k = KernelFunction(gamma=gamma)
	estimator = MBRPositiveEstimator(process, hierarchical_structure, kernel_object=k,
									 B=B, m=m, d=d, embedding=embedding, basis="custom")
	min_vol, max_vol = estimator.get_min_max()

	dt = 10. / (b * min_vol)
	dt = dt * 2

	print("Suggested dt:", dt)
	c = ['k', 'r', 'b', 'y', 'g', 'orange', 'brown', 'purple'] + ['k' for i in range(500)]

	no_sets = len(Sets)
	no_samples = 0
	data = []
	samples = []
	repeats = 2

	for i in range(no_samples):
		j = np.random.randint(0, no_sets, 1)
		S = Sets[j[0]]
		for _ in range(repeats):
			sample = process.sample_discretized(S, dt)
			samples.append(sample)
			data.append((S, sample, dt))

	sample_D = process.sample_discretized(D, dt)
	samples.append(sample_D)
	no_samples = repeats * no_samples + 1
	data.append((D, sample_D, dt))

	estimator.load_data(data)

	xtest = D.return_discretization(n=n)

	# likelihood based
	estimator.penalized_likelihood()
	rate_mean = estimator.mean_rate(D, n=n)

	# _, lcb, ucb = estimator.map_lcb_ucb(D, n, beta=2.)

	for j in range(no_samples):
		if samples[j] is not None:
			plt.plot(samples[j], samples[j] * 0, 'o', color=c[j])

	plt.plot(xtest, rate_mean, label='likelihood - locations known')
	# plt.fill_between(xtest.numpy().flatten(), lcb.numpy().flatten(), ucb.numpy().flatten(), alpha=0.4,
	#				 color='blue', label='triangle')
	process.visualize(D, samples=0, n=n, dt=1.)
