import cvxpy as cp
import matplotlib.pyplot as plt
import mosek
import numpy as np
import torch

from stpy.borel_set import BorelSet, HierarchicalBorelSets
from stpy.kernels import KernelFunction
from stpy.point_processes.poisson import PoissonPointProcess
from stpy.point_processes.poisson_rate_estimator import PoissonRateEstimator


class LogLinearRateEstimator(PoissonRateEstimator):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def least_squares_weighted(self, threads=0):
		theta = cp.Variable(self.get_m())

		mask = self.bucketized_counts.clone().numpy() > 0

		observations = self.total_bucketized_obs[mask].clone().numpy()
		phis = self.varphis[mask, :].clone().numpy()
		tau = self.total_bucketized_time.clone().numpy()

		variances = self.variances.view(-1).clone().numpy()

		for i in range(variances.shape[0]):
			if mask[i] > 0:
				variances[i] = variances[i] * tau[i] * self.variance_correction(variances[i] * tau[i])

		selected_variances = variances[mask]
		print(np.log(observations))
		print(selected_variances)
		objective = cp.Minimize(
			cp.sum_squares((phis @ theta) - np.log(observations) / tau[mask]))  # + self.s * cp.norm2(theta))

		prob = cp.Problem(objective)

		prob.solve(solver=cp.MOSEK, warm_start=False, verbose=True,
				   mosek_params={mosek.iparam.num_threads: threads})

		self.rate = torch.from_numpy(theta.value)
		print(self.rate)
		return self.rate

	def mean_var_reg_set(self, S, dt=1., beta=2.):
		if self.approx_fit == False:
			self.W = self.construct_covariance_matrix_regression()
			self.approx_fit = True

		map = 0
		lcb = 0
		ucb = 0
		for set in self.basic_sets:
			if S.inside(set):
				x = self.packing.integral(set).view(-1, 1)
				lcb = lcb + torch.exp(dt * (x @ self.rate - beta * np.sqrt(x.T @ self.W_inv @ x)))
				ucb = ucb + torch.exp(dt * (x @ self.rate + beta * np.sqrt(x.T @ self.W_inv @ x)))
				map = map + torch.exp(dt * x @ self.rate)
		return map, ucb, lcb

	def fit_ellipsoid_approx(self):
		self.W = self.construct_covariance_matrix_regression()
		self.W_inv = torch.pinverse(self.W)

	# def map_lcb_ucb_approx_action(self, S, dt=1., beta=2.):
	# 	phi = self.packing.integral(S) * dt
	# 	map = phi @ self.rate
	# 	ucb = map + beta * np.sqrt(phi @ self.W_inv_approx @ phi.T)
	# 	ucb = np.minimum(ucb, self.B * S.volume() * dt)
	#
	# 	lcb = map - beta * np.sqrt(phi @ self.W_inv_approx @ phi.T)
	# 	lcb = np.maximum(lcb, self.b * S.volume() * dt)
	# 	return map, lcb, ucb

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
						W = W + A / (variances[index_o])

			elif self.feedback == "histogram":

				for datapoint in self.data:
					(S, obs, dt) = datapoint
					varphi = self.packing.integral(S) * dt
					variance = varphi @ self.rate
					variance = variance
					A = varphi.view(-1, 1) @ varphi.view(1, -1)
					W = W + A / variance

		return W + torch.eye(self.get_m()).double() * self.s

	def mean_set(self, S, dt=1.):
		mu = 0
		for set in self.basic_sets:
			if S.inside(set):
				mu = mu + torch.exp(dt * self.packing.integral(set) @ self.rate)
		return mu


if __name__ == "__main__":
	torch.manual_seed(2)
	np.random.seed(2)
	d = 1
	gamma = 0.1
	n = 64
	B = 4.
	b = 0.1

	process = PoissonPointProcess(d=1, B=B, b=b)
	Sets = []
	levels = 5
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	Sets = hierarchical_structure.get_all_sets()

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	m = 128
	k = KernelFunction(gamma=gamma)
	estimator = LogLinearRateEstimator(process, hierarchical_structure,
									   kernel_object=k, B=B, m=m, d=d, estimator='least-sq')

	min_vol, max_vol = estimator.get_min_max()

	dt = 1. / (b * min_vol)
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
	estimator.fit_gp()

	for set in estimator.basic_sets:
		x = np.linspace(set.bounds[0, 0], set.bounds[0, 1], 2)
		val = estimator.mean_set(set)
		plt.plot(x, x * 0 + float(val), 'b-o')
		vol = process.rate_volume(set)
		plt.plot(x, x * 0 + float(vol), '-o', color='orange')
	for j in range(no_samples):
		if samples[j] is not None:
			plt.plot(samples[j], samples[j] * 0, 'o', color=c[j])

	process.visualize(D, samples=0, n=n, dt=1.)
