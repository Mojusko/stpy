from stpy.estimator import Estimator
from stpy.helpers.helper import *
from stpy.kernels import KernelFunction


class GammaContProcess(Estimator):

	def __init__(self, gamma=1, s=0.001, kappa=1., kernel="squared_exponential", diameter=1.0,
				 groups=None, bounds=None, nu=2, safe=False, kernel_custom=None, d=1):
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
		self.safe = False
		self.fit = False
		self.diameter = diameter
		self.bounds = bounds
		self.admits_first_order = False
		self.back_prop = True

		## kernel hyperparameters
		if kernel_custom is not None:
			self.kernel_object = kernel_custom
			self.kernel = kernel_custom.kernel
		else:
			self.kernel_object = KernelFunction(kernel_name=kernel, gamma=gamma, nu=nu, groups=groups, kappa=kappa)
			self.kernel = self.kernel_object.kernel

			self.gamma = gamma
			self.v = nu
			self.groups = groups
			self.kappa = kappa
			self.custom = kernel_custom
			self.optkernel = kernel

	def description(self):
		"""
		Description of GP in text
		:return: string with description
		"""
		return self.kernel_object.description() + "\n noise: " + str(self.s)

	def get_gamma(self, t):
		"""
		??
		:param t:
		:return:
		"""
		if self.optkernel == "squared_exponential" and self.groups is None:
			return (np.log(t)) ** self.d
		elif self.optkernel == "linear":
			return 10 * self.d
		elif self.optkernel == "squared_exponential" and self.groups is not None:
			return len(self.groups) * (np.log(t))
		elif self.optkernel == "matern":
			return (np.log(t)) ** self.d
		elif self.optkernel == "modified_matern":
			return (np.log(t)) ** self.d

	def make_safe(self, x):
		"""
		Make the input dataset numerically stable by removing duplicates?
		:param x:
		:return:
		"""
		self.epsilon = 0.001
		# remove vectors that are very close to each other
		return x

	def fit_gp(self, x, y, iterative=False, extrapoint=False):
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
				self.n, self.d = list(x.size())
			except:
				self.n, self.d = x.shape
			self.K = self.kernel(x, x) + self.s * self.s * torch.eye(self.n, dtype=torch.float64)

			self.fit = True
		else:
			# iterative inverse
			if (iterative == True):
				if extrapoint == False:
					last_point = self.x[-1, :].view(1, -1)
				else:
					last_point = x
				old_K = self.K
				old_Kinv = self.Kinv
			else:
				pass

		return None

	def beta(self, delta=1e-12, norm=1):
		beta_value = self.s * norm + torch.sqrt(
			2 * torch.log(1. / delta + torch.log(torch.det(self.K) / self.s ** self.n)))
		return beta_value

	def execute(self, xtest):
		if self.fit == True:
			K_star = self.kernel(self.x, xtest)
		else:
			K_star = None
		K_star_star = self.kernel(xtest, xtest)
		return (K_star, K_star_star)

	# @check_numpy(1)
	def mean_var(self, xtest, full=False):
		"""
		Return posterior mean and variance as tuple
		:param xtest: grid, numpy array (2D)
		:param full: Instead of just poinwise variance, full covariance can be outputed (bool)
		:return: (tensor,tensor)
		"""

		(K_star, K_star_star) = self.execute(xtest)

		if self.fit == False:
			if full == False:

				x = torch.sum(xtest, dim=1)
				first = torch.diag(K_star_star).view(-1, 1)
				variance = first
				yvar = torch.sqrt(variance)
			else:
				first = K_star_star
				yvar = first

			return (0 * x.view(-1, 1), yvar)

		if self.back_prop == False:
			decomp = torch.btrifact(self.K.unsqueeze(0))
			A = torch.btrisolve(self.y.unsqueeze(0), *decomp)[0, :, :]
			self.B = torch.t(torch.btrisolve(torch.t(K_star).unsqueeze(0), *decomp)[0, :, :])
		else:
			A, _ = torch.gesv(self.y, self.K)
			self.B = torch.t(torch.gesv(torch.t(K_star), self.K)[0])

		ymean = torch.mm(K_star, A)

		if full == False:
			first = torch.diag(K_star_star).view(-1, 1)
			second = torch.einsum('ij,ji->i', (self.B, torch.t(K_star))).view(-1, 1)
			variance = first - second
			yvar = torch.sqrt(variance)
		else:
			first = K_star_star
			second = torch.mm(self.B, torch.t(K_star))
			yvar = first - second

		return (ymean, yvar)

	def sample(self, xtest, size=1):
		"""
		Samples Path from GP, return a numpy array evaluated over grid
		:param xtest: grid
		:param size: number of samples
		:return: numpy array
		"""
		nn = list(xtest.size())[0]

		if self.fit == True:
			(ymean, yvar) = self.mean_var(xtest, full=True)
			Cov = yvar + self.s * self.s * torch.eye(nn, dtype=torch.float64)
			L = torch.cholesky(Cov, upper=False)
			random_vector = torch.normal(mean=torch.zeros(nn, size, dtype=torch.float64), std=1.)
			f = ymean + torch.abs(torch.mm(L, random_vector))
		else:
			(K_star, K_star_star) = self.execute(xtest)
			L = torch.cholesky(K_star_star + (10e-10 + self.s * self.s) * torch.eye(nn, dtype=torch.float64),
							   upper=False)
			random_vector = torch.normal(mean=torch.zeros(nn, size, dtype=torch.float64), std=1.)
			f = self.mu + torch.mm(L, random_vector)
		return f

	def sample_and_max(self, xtest, size=1):
		"""
		Samples Path from GP and takes argmax
		:param xtest: grid
		:param size: number of samples
		:return: (argmax, max)
		"""
		f = self.sample(xtest, size=size)
		self.temp = f
		val, index = torch.max(f, dim=0)
		return (xtest[index, :], val)
