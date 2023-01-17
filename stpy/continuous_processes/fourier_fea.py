from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.embeddings.embedding import *
from stpy.helpers.helper import *
from stpy.kernels import KernelFunction


class GaussianProcessFF(KernelizedFeatures):
	'''	
		Random Fourier Features for Gaussian Kernel
	'''

	def __init__(self, project=None, gamma=0.1, s=0.001, approx="rff", m=100, d=1, diameter=1.0, verbose=True,
				 groups=None,
				 bounds=None, scale=1.0, kernel="squared_exponential", nu=0.5, kappa=1.0):

		self.gamma = gamma
		self.s = s
		self.x = None
		self.K = 0
		self.mu = 0.0
		self.fit = False
		self.beta = None
		self.m = m
		self.project = None
		self.nu = nu
		self.lam = 1.
		if groups is None:
			self.no_groups = 1
		else:
			self.no_groups = len(groups)

		self.approx = approx
		self.d = d
		self.bounds = bounds
		self.groups = groups
		self.diameter = diameter
		self.admits_first_order = True
		self.verbose = verbose
		self.kernel = kernel
		self.scale = scale
		self.m_old = None
		self.kappa = kappa
		self.heuristic_variance = False
		if self.groups is None:
			self.embedding_map = self.sample_embedding(self.d, self.m, self.gamma)
			self.m = self.embedding_map.m
		else:
			self.no_groups = float(len(self.groups))
			self.embedding_map = self.sample_embedding_group()

	def resample(self):
		self.embedding_map = self.sample_embedding_group()

	def description(self):
		"""
		Description of GP in text
		:return: string with description
		"""
		return "Fourier Features object\n" + "Appprox: " + self.approx + "\n" + "Bandwidth: " + str(
			self.gamma) + "\n" + "Groups:" + str(self.groups) + "\n noise: " + str(self.s)

	def get_gamma(self, t):
		if self.kernel == "squared_exponential" and self.groups is None:
			return (np.log(t)) ** self.d
		elif self.kernel == "linear":
			return 10 * self.m
		elif self.kernel == "squared_exponential" and self.groups is not None:
			return len(self.groups) * (np.log(t))
		elif self.kernel == "matern":
			return (np.log(t)) ** self.d
		elif self.kernel == "modified_matern":
			return (np.log(t)) ** self.d

	def sample_embedding_group(self):
		# self.m is a vector of ms
		# self.gamma is a vector of gammas
		embedding_map = []

		self.d_effective = int(self.d / self.no_groups)

		if self.groups is not None:
			self.d_group_sizes = [len(group) for group in self.groups]
			self.d_effective = max(self.d_group_sizes)

		if np.sum(np.array(list(self.gamma.size()))) > 1:
			self.gamma = self.gamma
		else:
			self.gamma = torch.ones(int(self.no_groups), dtype=torch.float64) * self.gamma

		for i, group in enumerate(self.groups):
			embedding_map.append(self.sample_embedding(len(group), self.m[i], self.gamma[i]))
			self.m[i] = embedding_map[i].m
		return embedding_map

	def sample_embedding(self, d_effective, m, gamma):
		if self.m_old is not None:
			self.m = self.m_old

		if self.approx == "quad":
			embedding_map = QuadratureEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
												groups=None,
												kernel=self.kernel, approx=self.approx)
		elif self.approx == "rff":
			embedding_map = RFFEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
										 groups=None,
										 kernel=self.kernel, approx=self.approx)
		elif self.approx == "rff2":
			embedding_map = RFFEmbedding(biased=True, gamma=gamma, nu=self.nu, m=m, d=d_effective,
										 diameter=self.diameter, groups=None,
										 kernel=self.kernel, approx=self.approx)
		elif self.approx == "halton":
			embedding_map = RFFEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
										 groups=None,
										 kernel=self.kernel, approx=self.approx)
		elif self.approx == "hermite":
			embedding_map = HermiteEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
											 groups=None,
											 kernel=self.kernel, approx=self.approx)
		elif self.approx == "trapezoidal":
			embedding_map = TrapezoidalEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
												 groups=None,
												 kernel=self.kernel, approx=self.approx)
		elif self.approx == "ccff":
			embedding_map = ClenshawCurtisEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
													groups=None,
													kernel=self.kernel, approx=self.approx)
		elif self.approx == "matern_secific":
			embedding_map = MaternEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
											groups=None,
											kernel=self.kernel, approx=self.approx)
		elif self.approx == "quad_periodic":
			embedding_map = QuadPeriodicEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
												  groups=None,
												  kernel=self.kernel, approx=self.approx)
		elif self.approx == "kl":
			embedding_map = KLEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective,
										diameter=self.diameter, groups=None, kernel=self.kernel, approx=self.approx)
		elif self.approx == "orf":
			embedding_map = RFFEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
										 groups=None,
										 kernel=self.kernel, approx=self.approx)
		else:
			embedding_map = QuadratureEmbedding(gamma=gamma, nu=self.nu, m=m, d=d_effective, diameter=self.diameter,
												groups=None,
												kernel=self.kernel, approx=self.approx)
		self.m_old = self.m

		return embedding_map

	def embed(self, x):
		if self.groups is None:

			if self.project is not None:
				x = self.project(x)

			return self.embedding_map.embed(x)

		else:
			return self.embed_whole(x)

	def embed_group(self, x, group):
		return self.embedding_map[group].embed(x) / (np.sqrt(self.no_groups))

	def embed_whole(self, x):
		if self.project is not None:
			x = self.project(x)

		if self.groups == None:
			return self.embed(x)
		else:
			n = x.size()[0]
			M = torch.zeros(int(torch.sum(self.m)), n, dtype=torch.float64)
			for i, group in enumerate(self.groups):
				embeding = self.embed_group(x[:, group], i)
				index = int(torch.sum(self.m[0:i], dim=0))
				index_next = int(torch.sum(self.m[0:i + 1], dim=0))
				M[index:index_next, :] = torch.t(embeding)
		return torch.t(M)

	def get_basis_size(self):
		return self.m

	def set_basis_size(self, m):
		self.m_old = None
		self.m = m

	def right_kernel(self):
		embeding = self.embed(self.x)
		Z = self.linear_kernel(embeding, embeding)
		K = (Z + self.s * self.s * torch.eye(self.n, dtype=torch.float64))
		return K

	def fit_gp(self, x, y, iterative=False):
		'''
			Function to Fit GP
		'''

		self.x = x
		self.y = y
		self.n = list(self.x.size())[0]
		self.linear_kernel = KernelFunction(kernel_name="linear").linear_kernel

		if self.groups == None:
			embeding = self.embed(x)
			self.Z_ = self.linear_kernel(torch.t(embeding), torch.t(embeding))
			self.K = (self.Z_ + self.s * self.s * torch.eye(self.m, dtype=torch.float64))
			self.Q = torch.t(embeding)

		else:  ## additive models
			M = torch.t(self.embed_whole(x))
			self.Q = M
			self.Z_ = self.linear_kernel(M, M)
			self.K = self.kappa * self.Z_ + self.s * self.s * torch.eye(int(torch.sum(self.m)), dtype=torch.float64)

		self.fit = True

		return None

	def log_marginal_likelihood_self(self):
		return self.log_marginal_likelihood(self.gamma, torch.eye(self.d, dtype=torch.float64), self.kappa)

	def log_marginal_likelihood(self, gamma, Rot, kappa, kernel="default"):
		"""
		Calculated the log marginal likelihood
		:param kernel: custom kenrel object
		:return: float
		"""
		# func = self.kernel_object.get_kernel_function()

		self.x = torch.mm(self.x, Rot)
		L = torch.torch.cholesky(self.K, upper=False)
		logdet = -0.5 * 2 * torch.sum(torch.log(torch.diag(L)))

		Q = self.embed_whole(self.x)
		rhs = torch.mm(torch.t(Q), self.y)
		alpha, _ = torch.solve(rhs, self.K)
		logprob = -0.5 * (torch.mm(torch.t(self.y), self.y) - torch.mm(torch.t(rhs),
																	   alpha)) / self.s ** 2 + logdet  # - 0.5*self.n*np.log(2*np.pi)
		logprob = -logprob

		return logprob

	def optimize_params(self, type="bandwidth", repeats=10, optimizer="pymanopt"):
		pass

	def mean_std(self, xtest, reuse=False):
		'''
			Calculate mean and variance for GP at xtest points
		'''
		# compute the mean at our test points.

		if self.project is not None:
			self.project(xtest)

		if self.groups == None:
			embeding = self.embed(xtest)
			Q = self.embed(self.x)
		else:
			self.Z_ = self.K - self.s * self.s * torch.eye(int(torch.sum(self.m)), dtype=torch.float64)
			embeding = self.embed_whole(xtest)
			Q = self.embed_whole(self.x)

		theta_mean, _ = torch.solve(torch.mm(torch.t(Q), self.y), self.K)
		ymean = torch.mm(embeding, theta_mean)

		temp = torch.t(torch.solve(torch.t(embeding), self.K)[0])
		diagonal = self.s * self.s * torch.einsum('ij,ji->i', (temp, torch.t(embeding))).view(-1, 1)
		yvar = torch.sqrt(diagonal)

		return (ymean, yvar)

	# def posterior_inf(self, xtest, tol=10e-5, max_int=20000):
	# 	alpha = np.random.randn(self.n, 1)
	# 	err = 10.
	# 	F = 10.0
	# 	counter = 0
	# 	embeding = self.embed(self.x)
	# 	K = (linear_kernel(embeding.T, embeding.T) + self.s * self.s * np.eye(self.n))
	# 	Kinv = np.linalg.pinv(K)
	#
	# 	q = []
	# 	for index in range(self.n):
	# 		q.append(self.embed(self.x[index, :].reshape(1, -1)))
	# 	q = np.array(q)
	#
	# 	while (counter < max_int and err / F > tol):
	# 		# first find which index gives maximum
	# 		# print (K.shape)
	# 		index = np.argmax(np.abs(K.dot(alpha) - self.y))
	# 		sign = np.sign(K.dot(alpha)[index] - self.y[index])
	#
	# 		k = linear_kernel(embeding.T, q[index, :, :].T).reshape(-1, 1)
	# 		# print ("k: ", k.shape)
	# 		oldalpha = alpha
	# 		alpha = alpha - 1. / np.sqrt(counter + 1) * Kinv.dot(self.s * K.dot(alpha) + sign * k)
	# 		err = np.linalg.norm(oldalpha - alpha)
	# 		counter += 1
	# 		F = np.max(np.abs(K.dot(alpha) - self.y)) + self.s * alpha.T.dot(K.dot(alpha))[0][0]
	#
	# 	y_inf = linear_kernel(self.embed(self.x).T, self.embed(xtest).T).T.dot(alpha)
	# 	return y_inf

	def sample_theta(self, size=1):
		if self.groups is None:
			basis = self.m
		else:
			basis = int(int(torch.sum(self.m)))
		zeros = torch.zeros(basis, size, dtype=torch.float64)
		random_vector = torch.normal(mean=zeros, std=1.)

		if self.fit == True:
			# random vector
			Z = torch.pinverse(self.K)
			self.L = torch.cholesky(Z, upper=False)
			theta_mean = torch.mm(Z, torch.mm(self.Q, self.y))
			theta = torch.mm(self.s * self.L, random_vector)
			theta = theta + theta_mean
		else:
			theta_mean = 0
			Z = (1. + self.s * self.s) * torch.eye(basis, dtype=torch.float64)
			L = torch.cholesky(Z, upper=False)
			theta = torch.mm(L, random_vector) + theta_mean
		return theta

	def sample(self, xtest, size=1):
		'''
			Sample functions from Gaussian Process
		'''
		theta = self.sample_theta(size=size)
		if self.groups == None:
			f = torch.mm(self.embed(xtest), theta)
		else:
			f = torch.zeros(xtest.size()[0], size, dtype=torch.float64)
			for i, group in enumerate(self.groups):
				embeding = self.embed_group(xtest[:, group], i)
				index = int(torch.sum(self.m[0:i], dim=0))
				index_next = int(torch.sum(self.m[0:i + 1], dim=0))
				f += torch.mm(embeding, theta[index:index_next, :])
		return f

	def sample_and_max(self, xtest, size=1):
		'''
			Sample functions from Gaussian Process and take Maximum
		'''
		f = self.sample(xtest, size=size)

		index = np.argmax(f.detach(), axis=0)
		return (xtest[index, :], f[index, :])

	def ucb_optimize(self, beta, multistart=25):

		mean = lambda x: self.mean_std(torch.from_numpy(x).view(1, -1))[0][0][0]
		sigma = lambda x: self.mean_std(torch.from_numpy(x).view(1, -1))[1][0][0]

		fun = lambda x: -(mean(x) + np.sqrt(beta) * sigma(x))
		# grad = lambda x: -complex_step_derivative(fun,1e-10,x.reshape(1,-1))

		mybounds = self.bounds
		results = []
		from scipy.optimize import minimize

		for i in range(multistart):
			x0 = np.random.randn(self.d)
			for i in range(self.d):
				x0[i] = np.random.uniform(mybounds[i][0], mybounds[i][1])

			res = minimize(fun, x0, method="L-BFGS-B", jac=None, tol=0.0001, bounds=mybounds)
			solution = res.x
			results.append([solution, -fun(solution)])

		results = np.array(results)
		index = np.argmax(results[:, 1])
		solution = results[index, 0]

		return (solution, -fun(solution))

	def special_embed_eval(self, x, theta):
		f = 0
		x = torch.from_numpy(x)
		# print (x)
		for i, group in enumerate(self.groups):
			embeding = self.embed_group(x[group].view(-1, len(group)), i)
			index = torch.sum(self.m[0:i], dim=0)
			index_next = torch.sum(self.m[0:i + 1], dim=0)
			f += torch.mm(embeding, theta[int(index):int(index_next), :])
		return f.numpy()

	def special_embed_eval_grad(self, x, theta):
		ff = lambda x: self.special_embed_eval(x.flatten(), theta)
		grad = complex_step_derivative(ff, 1e-10, x.reshape(-1, 1).T).flatten()
		return grad

	def get_lambdas_additive(self, theta):
		fun = lambda x: -self.special_embed_eval(x, theta)
		grad = lambda x: -self.special_embed_eval_grad(x, theta)
		return [fun, grad]

	def get_lambdas(self, theta):

		# complex step differentiation
		fun = lambda x: -(torch.mm(self.embed(torch.from_numpy(x).view(1, self.d)), theta).numpy()).flatten()
		grad = lambda x: -complex_step_derivative(fun, 1e-10, x.reshape(self.d, 1).T).flatten()
		return [fun, grad]

	def sample_and_optimize(self, xtest=None, multistart=25, minimizer="L-BFGS-B", grid=100, verbose=0):
		'''
			Sample functions from Gaussian Process and take Maximum using
			first order maximization
		'''

		# sample linear approximating
		theta = self.sample_theta()
		from scipy.optimize import minimize

		# get bounds
		if self.bounds == None:
			mybounds = tuple([(-self.diameter, self.diameter) for i in range(self.d)])
		else:
			mybounds = self.bounds

		fun = lambda x: -torch.mm(torch.t(theta), torch.t(self.embed(torch.from_numpy(x).view(1, -1)))).numpy()

		results = []
		for j in range(multistart):
			x0 = np.random.randn(self.d)
			for i in range(self.d):
				x0[i] = np.random.uniform(mybounds[i][0], mybounds[i][1])

			if minimizer == "L-BFGS-B":
				res = minimize(fun, x0, method="L-BFGS-B", jac=None, tol=0.0001, bounds=mybounds)
				solution = res.x
			elif minimizer == "ProjGD":
				res = projected_gradient_descent(fun, grad, x0, mybounds, tol=0.001,
												 nu=1. / (self.m * np.max(np.abs(theta))))
				solution = res.x
			elif minimizer == "coordinate-wise":

				solution = np.random.randn(self.d)
				for i in range(self.d):
					if verbose > 0:
						print("Dimension: ", i)
					fun_cw = lambda x: lambda_coordinate(fun, x0, i, x)
					ranges = [slice(mybounds[i][0], mybounds[i][1], 1. / float(grid))]
					out = scipy.optimize.brute(fun_cw, ranges, finish=None)
					solution[i] = out
				if verbose > 0:
					print("Soln:", out.T)
			elif minimizer == "CD_cw":
				raise BaseException("Not implemented yet")
			else:
				raise AssertionError("Wrong optimizer selected.")

			results.append([solution, -fun(solution)])

		results = np.array(results)
		index = np.argmax(results[:, 1])
		solution = results[index, 0]

		return (torch.from_numpy(solution), -torch.from_numpy(fun(solution)))


if __name__ == "__main__":
	# domain size
	L_infinity_ball = 1
	# dimension
	d = 2
	# error variance
	s = 0.001
	# grid density
	n = 50
	# number of intial points
	N = 200
	# smoothness
	gamma = torch.from_numpy(np.array([0.4, 0.4]))
	# test problem

	xtest = torch.from_numpy(interval(n, d))
	x = torch.from_numpy(np.random.uniform(-L_infinity_ball, L_infinity_ball, size=(N, d)))

	f_no_noise = lambda q: torch.sin(torch.sum(q * 4, dim=1)).view(-1, 1)
	# f_no_noise = lambda q: torch.sin((q[:,0] * 4)).view(-1, 1)

	f = lambda q: f_no_noise(q) + torch.normal(mean=torch.zeros(q.size()[0], 1, dtype=torch.float64), std=1.,
											   out=None) * s
	# targets
	y = f(x)

	# GP model with squared exponential
	m = torch.from_numpy(np.array([100, 100]))

	groups = [[0], [1]]
	GP = GaussianProcessFF(kernel="squared_exponential", s=s, m=m, d=d, gamma=gamma, groups=groups, approx="hermite")
	# GP2 = GaussianProcess(kernel="ard", s=s, d=d, gamma=gamma, groups=None)

	# fit GP
	GP.fit_gp(x, y)
	# GP2.fit_gp(x,y)

	GP.optimize_params("rots", 10, optimizer="pymanopt")

	print("Log probability:", GP.log_marginal_likelihood_self())
	# print ("Log probability:", GP2.log_marginal_likelihood_self() )

	GP.visualize(xtest, f_true=f_no_noise)
# GP2.visualize(xtest, f_true=f_no_noise)


# test kernel approximation capability
# s = 0.001
# print("%4s %2s %20s %20s %22s %22s %21s %21s" % (
# 	"m", "N", "max|K-K'|", "|K-K'|_2", "max|K^{-1}-K'^{-1}|", "|K^{-1}-K'^{-1}|_2", "|mu-mu'|_\infty", "theory"))
#
# for n in [2, 4, 8, 16, 32, 64]:
# 	TT = code.test_problems.test_functions.test_function()
# 	(d, xtest, x, gamma) = TT.f_bounds(n, 100, d=1, L_infinity_ball=1)
# 	gamma = 1
#
# 	f = lambda x: TT.f(x, sigma=s)
# 	y = f(x)
#
# 	for m in [2, 4, 8, 16, 32, 64, 128, 256]:
# 		GP = code.gps.gauss_procc.GaussianProcess(s=s, gamma=gamma,kernel = "modified_matern",nu = 4)
# 		GP2 = GaussianProcessFF(s=s, gamma=gamma, m=m, d=d,kernel="modified_matern",approx="quad", diameter=1, scale=1.,nu =4)
#
# 		GP.fit_GP(x, y)
# 		GP2.fit_GP(x, y)
#
# 		(mu, var) = GP.mean_var(xtest)
# 		(mu2, var2) = GP2.mean_var(xtest)
#
# 		max_discrepancy = np.max(np.abs(mu - mu2))
# 		from scipy.misc import factorial
#
# 		bar_m = lambda m: np.power(m, 1.)
#
# 		from scipy.misc import factorial
#
# 		theory_r = lambda m,diameter: (np.power(2, (d - 1))) * d * factorial(bar_m(m)) / (
# 					factorial(2 * bar_m(m)) * np.power(2, bar_m(m))) * np.power(diameter, 2 * bar_m(m)) * (
# 										 1 + n ** (2. / 3.) / (s * s))
#
# 		theory_discrepancy = theory_r(m, 2.)
#
# 		deviation = np.max(np.abs(GP2.embed(x).dot(GP2.embed(x).T) + GP2.s * GP2.s * np.eye(GP2.n) - GP.K))
# 		deviation_norm = np.linalg.norm(GP2.embed(x).dot(GP2.embed(x).T) + GP2.s * GP2.s * np.eye(GP2.n) - GP.K)
#
# 		deviation2 = np.max(np.abs(
# 			np.linalg.inv(GP2.embed(x).dot(GP2.embed(x).T) + GP2.s * GP2.s * np.eye(GP2.n)) - np.linalg.inv(GP.K)))
# 		deviation2_norm = np.linalg.norm(
# 			np.linalg.inv(GP2.embed(x).dot(GP2.embed(x).T) + GP2.s * GP2.s * np.eye(GP2.n)) - np.linalg.inv(GP.K))
#
# 		print("%4d %2d %20.15f %20.15f %22.15f %22.15f %21.15f %21.15f" % (
# 			m, n, deviation, deviation_norm, deviation2, deviation2_norm, max_discrepancy, theory_discrepancy))
#
#
