from stpy.regression.regularized_dictionary.regularized_dictionary import Regulariz
from stpy.kernel import KernelFunction


class GaussianProcess(Regulariz):

	def __init__(self, gamma=1,
                 s=0.001,
                 kappa=1., kernel_name="squared_exponential", diameter=1.0,
				 groups=None, bounds=None, nu=1.5, kernel=None, d=1, power=2, lam=1., loss = 'squared', huber_delta = 1.35,
				 hyper = 'classical', B = 1., svr_eps = 0.1, packet_size = 10000):
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
		self.lam = lam
		self.total_bound = B
		self.prob = 0.5
		self.svr_eps = svr_eps
		self.safe = False
		self.fitted = False
		self.diameter = diameter
		self.bounds = bounds
		self.admits_first_order = False
		self.back_prop = True
		self.loss = loss
		self.huber_delta = huber_delta
		self.hyper = hyper
		self.prepared_log_marginal = False
		self.warm_start_solution = None
		self.max_size = packet_size
		## kernel hyperparameters
		if kernel is not None:
			self.kernel_object = kernel
			self.kernel = kernel.kernel
			self.d = kernel.d
		else:
			self.kernel_object = KernelFunction(kernel_name=kernel_name, gamma=gamma, nu=nu, groups=groups, kappa=kappa,
												power=power, d=d)
			self.kernel = self.kernel_object.kernel

			self.gamma = gamma
			self.v = nu
			self.groups = groups
			self.kappa = kappa
			self.custom = kernel
			self.optkernel = kernel_name