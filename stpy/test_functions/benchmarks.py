import numpy as np
import torch

import stpy
from stpy.test_functions.swissfel_simulator import FelSimulator
from stpy.continuous_processes.gauss_procc import GaussianProcess


class BenchmarkFunction():

	def __init__(self, type="discrete", d=1, gamma=1.0, dts=None, **kwargs):
		self.scale = 1.0
		self.type = type
		self.gamma = gamma
		self.d = d
		self.dts = None
		self.groups = None

	def eval_noiseless(self, X):
		if X.size()[1] != self.d:
			raise AssertionError("Invalid dimension for the Benchmark function ...")
		pass

	def eval(self, X, sigma=None):
		z = self.eval_noiseless(X)
		if sigma is None:
			y = z/self.scale + self.s * torch.randn(X.size()[0], 1, dtype=torch.float64)
		else:
			y = z/self.scale + sigma * torch.randn(X.size()[0], 1, dtype=torch.float64)
		return y

	def optimum(self):
		return 1.0

	def maximum(self, xtest=None):
		if self.type == "discrete":
			self.max = self.maximum_discrete(xtest)
		else:
			self.max = self.maximum_continuous()
		return self.max

	def maximum_discrete(self, xtest):
		maximum =torch.max(self.eval_noiseless(xtest))
		return maximum

	def maximum_continuous(self):
		return 1.0

	def scale_max(self, xtest=None):
		self.scale = self.maximum(xtest=xtest)
		print("Scaling with", self.scale)

	def optimize(self, xtest, sigma, restarts=5):
		(n, d) = xtest.size()
		ytest = self.eval(xtest, sigma=sigma)
		kernel = stpy.kernels.KernelFunction(kernel_name="ard", gamma=torch.ones(d, dtype=torch.float64) * 0.1,
											 groups=self.groups)
		GP = stpy.continuous_processes.gauss_procc.GaussianProcess(kernel_custom=kernel, s=sigma, d=d)
		GP.fit(xtest, ytest)
		GP.optimize_params(type="bandwidth", restarts=restarts)
		print("Optimized")
		# GP.visualize(xtest)
		self.gamma = torch.min(kernel.gamma)
		return self.gamma

	def return_params(self):
		return (self.gamma, self.groups, self.d)

	def bandwidth(self):
		return self.gamma

	def set_group_param(self, groups):
		self.groups = groups

	def bounds(self):
		b = tuple([(-0.5, 0.5) for i in range(self.d)])
		return b

	def initial_guess(self, N, adv_inv=False):
		if adv_inv == False:
			x = torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(N, self.d)))
		else:
			x = torch.from_numpy(np.random.uniform(-0.5, 0., size=(N, self.d)))
		return x

	def interval(self, n, L_infinity_ball=0.5):
		if n == None:
			xtest = None
		else:
			xtest = torch.from_numpy(stpy.helpers.helper.interval(n, self.d, L_infinity_ball=L_infinity_ball))
		return xtest

	def visualize(self, xtest):
		import matplotlib.pyplot as plt
		d = xtest.size()[1]
		if d == 1:
			plt.figure(figsize=(15, 7))
			plt.clf()
			plt.plot(xtest.numpy(), self.eval_noiseless(xtest)[:, 0].numpy())
			plt.show()
		elif d == 2:
			from scipy.interpolate import griddata
			plt.figure(figsize=(15, 7))
			plt.clf()
			ax = plt.axes(projection='3d')
			xx = xtest[:, 0].numpy()
			yy = xtest[:, 1].numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z = griddata((xx, yy), self.eval_noiseless(xtest)[:, 0].numpy(), (grid_x, grid_y), method='linear')
			ax.plot_surface(grid_x, grid_y, grid_z, color='b', alpha=0.4)
			plt.show()


class CamelbackBenchmark(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.d = 2

	def eval_noiseless(self, X):
		super().eval_noiseless(X)
		xx = X[:, 0] * 4
		yy = X[:, 1] * 2
		y = (4. - 2.1 * xx ** 2 + (xx ** 4) / 3.) * (xx ** 2) + xx * yy + (-4. + 4 * (yy ** 2)) * (yy ** 2)
		y = -y.view(X.size()[0], 1)
		# y = np.tanh(y)
		y = y / 5.
		return y / self.scale


# def optimize(self,xtest,sigma, restarts = 5):
# self.gamma = 0.3

# self.gamma = 0.3
class QuadraticBenchmark(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.d = kwargs['d']
		self.type = "continuous"

		if 'R' in kwargs:
			self.R = kwargs['R']
			print("Quadratic Problem: Rotating - no longer additive.")
			print(self.R)
		else:
			self.R = torch.eye(self.d, self.d, dtype=torch.float64)
			print("Quadratic Problem: Additive.")

	def eval_noiseless(self, X):
		D = torch.diag(torch.Tensor([1., 2.]).double())
		super().eval_noiseless(X)
		(n, d) = X.size()
		X = X @ self.R
		sum_ = torch.sum((X @ D) ** 2, dim=1)
		print(sum_.size())
		return -sum_.view(-1, 1) / self.scale + 1

	def bandwidth(self):
		return 0.2


class PolynomialBenchmark(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.d = kwargs['d']
		self.type = "continuous"

		if 'R' in kwargs:
			self.R = kwargs['R']
			print("Quadratic Problem: Rotating - no longer additive.")
			print(self.R)
		else:
			self.R = torch.eye(self.d, self.d, dtype=torch.float64)
			print("Quadratic Problem: Additive.")

	def eval_noiseless(self, X):
		D = torch.diag(torch.Tensor([1., 2.]).double())
		super().eval_noiseless(X)
		(n, d) = X.size()
		X = X @ self.R
		sum_ = torch.sum((X @ D) ** 2, dim=1) + torch.sum((X @ D) ** 3, dim=1) * 0.5 + torch.sum((X @ D) ** 4, dim=1)
		print(sum_.size())
		return -sum_.view(-1, 1) / self.scale + 1

	def bandwidth(self):
		return 0.2


class MichalBenchmark(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.d = kwargs['d']
		self.type = "continuous"

		if 'R' in kwargs:
			self.R = kwargs['R']
			print("Michal Problem: Rotating - no longer additive.")
			print(self.R)
		else:
			self.R = torch.eye(self.d, self.d, dtype=torch.float64)
			print("Michal Problem: Additive.")

	def eval_noiseless(self, X):
		super().eval_noiseless(X)
		(n, d) = X.size()
		X = X @ self.R
		X = X / 0.75
		X = (X + 0.5) * np.pi
		ar = torch.from_numpy(np.arange(1, d + 1, 1, dtype=np.float64))
		sum_ = torch.sin(X) * torch.pow(torch.sin(ar * X / np.pi), int(2 * d))
		sum_ = torch.sum(sum_, dim=1).view(-1, 1)
		return sum_ / self.scale

	def optimize(self, xtest, sigma, restarts=5, n=512):
		xtest = torch.zeros(n, self.d, dtype=torch.float64)
		xtest[:, 0] = torch.linspace(-0.5, 0.5, n, dtype=torch.float64)
		ytest = self.eval(xtest, sigma=sigma)
		kernel = stpy.kernels.KernelFunction(kernel_name="ard", gamma=torch.ones(self.d, dtype=torch.float64) * 0.1,
											 groups=self.groups)
		GP = GaussianProcess(kernel=kernel, s=sigma, d=self.d)
		GP.fit_gp(xtest, ytest)
		#GP.optimize_params(type="bandwidth", restarts=restarts)
		#print("Optimized")
		#GP.back_prop
		self.gamma = torch.min(kernel.gamma)
		return self.gamma

	def bandwidth(self):
		return 0.2

	def maximum_continuous(self):
		opt = np.ones(shape=(20))
		# holds with different constnat
		opt[0] = 2.93254
		opt[1] = 2.34661
		opt[2] = 1.64107
		opt[3] = 1.24415
		opt[4] = 0.999643
		opt[5] = 0.834879
		opt[6] = 2.1089
		opt[7] = 1.84835
		opt[8] = 1.64448
		opt[9] = 1.48089
		opt[10] = 1.34678
		opt[11] = 1.2349
		opt[12] = 1.89701
		opt[13] = 1.76194
		opt[14] = 1.64477
		opt[15] = 1.54218
		opt[16] = 1.45162
		opt[17] = 1.37109
		opt[18] = 1.81774
		return float(opt[self.d])


class StybTangBenchmark(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.d = kwargs['d']
		self.type = "discrete"
		if 'R' in kwargs:
			self.R = kwargs['R']
			print("Stybtang Problem: Rotating - no longer additive.")
			print(self.R)
		else:
			self.R = torch.eye(self.d, self.d, dtype=torch.float64)
			print("Stybtang Problem: Additive")

	def eval_noiseless(self, X):
		super().eval_noiseless(X)
		(n, d) = X.size()
		X = X @ self.R
		X = X * 8
		Y = X ** 2
		sum_ = torch.sum(Y ** 2 - 16. * Y + 5 * X, dim=1).view(-1, 1)
		return -(0.5 * sum_ / (d * 200.) + 0.5)/self.scale

	# def maximum_continuous(self):
	# 	opt = np.ones(shape=(self.d)) * (-2.9035) / 8
	# 	opt = torch.from_numpy(opt.reshape(1, -1))
	# 	value = self.eval_noiseless(opt)[0][0] * 16
	# 	return value
	#
	# def optimize(self, xtest, sigma, restarts=5, n=512):
	# 	xtest = torch.zeros(n, self.d, dtype=torch.float64)
	# 	xtest[:, 0] = torch.linspace(-0.5, 0.5, n, dtype=torch.float64)
	# 	ytest = self.eval(xtest, sigma=sigma)
	# 	kernel = stpy.kernels.KernelFunction(kernel_name="ard", gamma=torch.ones(self.d, dtype=torch.float64) * 0.1,
	# 										 groups=self.groups)
	# 	GP = GaussianProcess(kernel_custom=kernel, s=sigma, d=self.d)
	# 	GP.fit(xtest, ytest)
	# 	GP.optimize_params(type="bandwidth", restarts=restarts)
	# 	print("Optimized")
	# 	self.gamma = torch.min(kernel.gamma)
	# 	return self.gamma

class GeneralizedAdditiveOverlap(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.d = kwargs['d']
		self.type = "continuous"

	def eval_noiseless(self, X):
		super().eval_noiseless(X)
		(n, d) = X.size()
		sum_ = torch.sum(torch.exp(-(torch.from_numpy(np.diff(X.numpy(), axis=1) / 0.25)) ** 2), dim=1).view(-1, 1)
		return 0.5 * sum_ / self.scale

	def maximum_continuous(self):
		opt = torch.from_numpy(np.zeros(shape=(1, self.d)))
		value = self.eval_noiseless(opt)[0][0]
		return value

	def optimize(self, xtest, sigma, restarts=5, n=512):
		xtest = torch.zeros(n, self.d, dtype=torch.float64)
		xtest[:, 0] = torch.linspace(-0.5, 0.5, n, dtype=torch.float64)
		ytest = self.eval(xtest, sigma=sigma)
		kernel = stpy.kernels.KernelFunction(kernel_name="ard", gamma=torch.ones(self.d, dtype=torch.float64) * 0.1,
											 groups=self.groups)
		GP = stpy.continuous_processes.gauss_procc.GaussianProcess(kernel_custom=kernel, s=sigma, d=self.d)
		GP.fit(xtest, ytest)
		GP.optimize_params(type="bandwidth", restarts=restarts)
		print("Optimized")
		# self.gamma = torch.min(kernel.gamma)
		# self.gamma = torch.zeros(1,1,dtype = torch.DoubleTensor)
		# self.gamma[0,0] =0.35
		self.gamma = torch.Tensor([0.35]).double()
		return self.gamma


class SwissFEL(BenchmarkFunction):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.d = kwargs['d']
		name = kwargs['dts']
		self.Simulator = FelSimulator(self.d, 0.0, "quadrupoles_2d")
		self.Simulator.load_fresh(name, dts='0')
		#self.groups = stpy.helpers.helper.full_group(self.d)
		GP = GaussianProcess(kernel_name="ard", d = self.d)
		self.Simulator.fit_simulator(GP, optimize="bandwidth", restarts=2)
		self.type = "continuous"
		self.s = self.Simulator.s

	def eval_noiseless(self, X):
		super().eval_noiseless(X)
		y = self.Simulator.eval(X, sigma=0)
		return y

	def maximum(self, xtest=None):
		return torch.max(self.Simulator.eval(xtest,sigma = 0))


class CustomBenchmark(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		if 'func' in kwargs:
			self.eval_f = kwargs['func']
		else:
			self.eval_f = lambda x: x[:, 0].view(-1, 1) * 0
		if 'likelihood' in kwargs:
			self.likelihood = kwargs['likelihood']
		else:
			self.likelihood = None

	def set_eval(self, f, scale=1.):
		self.eval_f = f
		self.scale = scale

	def eval_noiseless(self, X):
		#super().eval_noiseless(X)
		y = self.eval_f(X)
		return y / self.scale

	def eval(self, X):
		if self.likelihood is not None:
			return self.eval_noiseless(X)+self.likelihood.sample_noise(X)
		else:
			return self.eval_noiseless(X)

class GaussianProcessSample(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__()
		self.d = kwargs['d']
		self.kernel_name = kwargs['name']
		self.gamma = kwargs['gamma']
		self.sigma = kwargs['sigma']
		self.n = kwargs['n']
		self.sample(self.n)

	def sample(self, n):
		self.xtest = self.interval(n)
		GP = stpy.continuous_processes.gauss_procc.GaussianProcess(s=self.sigma, gamma=self.gamma,
																   kernel=self.kernel_name)
		self.sample = GP.sample(self.xtest).numpy()

	def isin(self, element, test_elements, assume_unique=False):
		(n, d) = element.shape
		(m, d) = test_elements.shape
		maskFull = np.full((n), False, dtype=bool)
		for j in range(m):
			mask = np.full((n), True, dtype=bool)
			for i in range(d):
				mask = np.logical_and(mask, np.in1d(element[:, i], test_elements[j, i], assume_unique=assume_unique))
			# mask = np.logical_and(mask, np.isclose(element[:, i], test_elements[j, i], atol=1e-02))
			# print (j, i, mask)
			maskFull = np.logical_or(mask, maskFull)
		# print (maskFull)
		return maskFull

	def eval_noiseless(self, X):
		super().eval_noiseless(X)
		mask = self.isin(self.xtest.numpy(), X.numpy())
		y = torch.from_numpy(self.sample[mask, :]).view(-1, 1)
		return y / self.scale

	def initial_guess(self, N, adv_inv=False):
		x = self.xtest[np.random.permutation(np.arange(0, self.xtest.size()[0], 1))[0:N], :]
		x = torch.sort(x, dim=0)[0]
		return x

	def scale_max(self, xtest=None):
		pass

	def optimize(self, xtest, sigma, restarts=5):
		pass


class KernelizedSample(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__()
		self.d = kwargs['d']
		# self.kernel_name = kwargs['name']
		# self.gamma = kwargs['gamma']
		self.sigma = kwargs['sigma']
		# self.n = kwargs['n']
		self.embed = kwargs['embed']
		self.m = kwargs['m']
		self.sample()

	def set_theta(self, theta):
		self.theta = theta

	def set_cutoff(self, cutoff):
		self.theta[cutoff:, 0] = 0

	def sample(self):
		print("basis size:", self.m)
		GP = stpy.continuous_processes.kernelized_features.KernelizedFeatures(d=self.d, m=self.m, embeding=self.embed)
		self.theta = GP.sample_theta(size=1)
		print(self.theta)

	def eval_noiseless(self, X):
		super().eval_noiseless(X)
		y = torch.mm(self.embed(X), self.theta)
		return y / self.scale

	def scale_max(self, xtest=None):
		pass

	def optimize(self, xtest, sigma, restarts=5):
		pass


class Simple1DFunction(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__()
		self.d = kwargs['d']

	def eval_noiseless(self, X):
		super().eval_noiseless(X)
		z = (X+0.5)*1.2
		y = -(1.4-3*z)*torch.sin(18*z)
		return y

	def maximum(self, xtest):
		return torch.max(torch.abs(self.eval_noiseless(xtest)))

class MultiRKHS(BenchmarkFunction):

	def __init__(self, **kwargs):
		super().__init__()
		self.d = 1

	def eval_noiseless(self, X):
		y = 10 * X ** 2  # + 0.1*torch.sin(10*X) #+ torch.sum(torch.exp(-(X-Xi)**2)*Wi)
		return y

	def maximum(self, xtest=None):
		pass


class LinearBenchmark(BenchmarkFunction):

	def __init__(self, d, s):
		self.d = d
		self.s = s
		# sample a plane
		self.theta = torch.randn(d, 1, dtype=torch.float64)

	def eval_noiseless(self, X):
		y = torch.mm(X, self.theta)
		return y

	def eval(self, X, sigma=None):
		if sigma is None:
			sigma = self.s
		z = self.eval_noiseless(X)
		y = z + sigma * torch.randn(X.size()[0], 1, dtype=torch.float64)
		return y
