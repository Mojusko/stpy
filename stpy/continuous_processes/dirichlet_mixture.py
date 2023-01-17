import matplotlib.pyplot as plt

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.embeddings.embedding import *
from stpy.estimator import Estimator


class DirichletMixture(Estimator):

	def __init__(self, processes):
		self.processes = processes
		self.k = len(self.processes)
		self.s = processes[0].s

	def fit_GP(self, X, y, xtest=None, N=200):
		self.X = X
		self.y = y
		n = X.shape[0]
		self.fit = True
		return True

	def custom_kernel(self, a, b, alpha):
		kernel = alpha[0] * self.processes[0].kernel(a, b)
		for j in np.arange(1, self.k, 1):
			kernel = kernel + alpha[j] * self.processes[j].kernel(a, b)
		return kernel

	def mean_var(self, xtest, N=100):

		self.K_mix = np.zeros(shape=(n, n))

		mu = xtest * 0
		s = xtest * 0

		samples = np.zeros(shape=(N, xtest.shape[0], xtest.shape[1]))

		for i in range(N):
			alpha = np.random.dirichlet(np.ones(shape=(self.k)) * (1. / float(self.k)), 1)[0]
			print("Dirichlet sample:", alpha)
			kernel = lambda a, b: self.custom_kernel(a, b, alpha)
			GP_mix = GaussianProcess(kernel="custom", custom=kernel, s=self.s)
			GP_mix.fit_GP(self.X, self.y)
			samples[i, :, :] = GP_mix.sample(xtest)

		mu = np.mean(samples, axis=0)
		s = np.var(samples, axis=0)
		s = np.sqrt(s)

		return (mu, s)

	def sample(self, xtest, size=1, with_mask=False):
		# sample a GP
		if self.fit == True:
			alpha = np.random.dirichlet(np.ones(shape=(self.k)) * (1. / float(self.k)), 1)[0]
			kernel = lambda a, b: self.custom_kernel(a, b, alpha)
			GP_mix = GaussianProcess(kernel="custom", custom=kernel, s=self.s)
			GP_mix.fit_GP(self.X, self.y)
			return GP_mix.sample(xtest)
		else:
			alpha = np.random.dirichlet(np.ones(shape=(self.k)) * (1. / float(self.k)), 1)[0]
			kernel = lambda a, b: self.custom_kernel(a, b, alpha)
			GP_mix = GaussianProcess(kernel="custom", custom=kernel, s=self.s)
			return GP_mix.sample(xtest)


if __name__ == "__main__":

	# domain size
	L_infinity_ball = 5
	# dimension
	d = 1
	# error variance
	s = 0.001
	# grid density
	n = 1024
	# number of intial points
	N = 15
	# smoothness
	gamma = 2

	# model
	GP1 = GaussianProcess(kernel="squared_exponential", s=s, gamma=1.5, diameter=L_infinity_ball)
	GP2 = GaussianProcess(kernel="squared_exponential", s=s, gamma=1.1)
	GP3 = GaussianProcess(kernel="modified_matern", s=s, kappa=1., nu=2, gamma=1.1)
	GP4 = GaussianProcess(kernel="linear", s=s, kappa=1.)

	# data
	# GPTrue = GaussianProcess(kernel="linear", s=0, kappa=1., diameter=L_infinity_ball)
	GPTrue = GaussianProcess(kernel="squared_exponential", s=s, gamma=2., kappa=1)
	# GPTrue = GaussianProcess(kernel = "modified_matern", s =s, kappa = 1., nu = 2, gamma = 1.1)

	# test environment
	TT = code.test_problems.test_functions.test_function()
	(d, xtest, x, gamma) = TT.sample_ss_bounds(N, n, d=d, L_infinity_ball=L_infinity_ball)
	f = lambda x: TT.sample_ss(x, sigma=0, GP=GPTrue)

	# targets
	y = f(x)
	GPs = [GP1, GP2, GP3, GP4]
	Mix = DirichletMixture(GPs)
	for j in range(N):
		plt.figure(1)
		plt.clf()
		X = x[0:j + 1, :].reshape(-1, 1)
		y = f(X)
		Mix.fit_GP(X, y)
		(mu, var) = Mix.mean_var(xtest)
		samples = Mix.sample(xtest, size=5)
		plt.plot(xtest, samples, '--', linewidth=3, alpha=0.1)
		plt.plot(xtest, mu, 'k', linewidth=4)
		plt.plot(xtest, mu, 'k', linewidth=4)
		plt.fill_between(xtest.flat, (mu - var).flat, (mu + var).flat, color="#dddddd")
		plt.plot(X, y, 'ro', markersize=10)
		plt.plot(xtest, f(xtest), 'g', linewidth=4)
		plt.draw()
		# plt.figure(2)
		# plt.clf()
		# plt.title("Probability of Category")
		# plt.bar(np.arange(len(GPs)), Mix.weights, np.ones(len(GPs))*0.5)
		# plt.xticks(np.arange(len(GPs)), [GP.description() for GP in GPs], rotation=30)
		# plt.subplots_adjust(bottom=0.35)
		# plt.draw()
		plt.pause(4)
