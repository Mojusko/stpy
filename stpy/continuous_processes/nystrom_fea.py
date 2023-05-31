import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.embeddings.embedding import *
from stpy.helpers.helper import *
from stpy.kernels import KernelFunction


class NystromFeatures(Embedding):
	'''
		Nystrom Features for Gaussian Kernel
	'''

	def __init__(self, kernel_object, m=100, approx="uniform", s=1., samples=100):

		self.fit = False
		self.m = m
		try:
			self.ms = int(torch.sum(m))
		except:
			self.ms = m
		self.samples = samples
		self.kernel_object = kernel_object
		self.kernel = kernel_object.kernel
		self.approx = approx
		self.s = s

	def description(self):
		"""
		Description of GP in text
		:return: string with description
		"""
		return "Nystrom\n" + "Appprox: " + self.approx

	def subsample(self, x, y):
		if self.approx == "uniform":
			C, w = self.uniform_subsampling(x, y)
		elif self.approx == "leverage":
			C, w = self.leverage_score_subsampling(x, y)
		elif self.approx == "online_leverage":
			C, w = self.sequential_leverage_score_subsampling(x, y)
		return (C, w)

	def uniform_subsampling(self, x, y):
		N = x.size()[0]
		C = np.random.choice(N, int(self.ms))
		weights = torch.ones(self.ms)
		return (C, weights)

	def leverage_score_subsampling(self, x, y):
		N = x.size()[0]
		from stpy.continuous_processes.gauss_procc import GaussianProcess
		GP = GaussianProcess(kernel_custom=self.kernel_object, s=self.s)
		GP.fit_gp(x, y)
		mean, leverage_scores = GP.mean_std(x)
		weights = torch.ones(self.ms)

		args = [0]
		size = 1

		for j in range(N):
			point = x[j, :]
			if size < self.ms:
				leverage_score = float(leverage_scores[j, :])
				q = np.random.binomial(self.ms, leverage_score)
				if q > 0:
					args.append(j)
					weights[size] = (q / float(self.ms)) / leverage_score
					size = size + 1
				else:
					pass

		print(args, weights)
		return (args, weights)

	def sequential_leverage_score_subsampling(self, x, y):
		N = x.size()[0]
		d = x.size()[1]
		from stpy.continuous_processes.gauss_procc import GaussianProcess
		GP = GaussianProcess(kernel_custom=self.kernel_object, s=self.s)

		dts = torch.zeros(self.ms, d, dtype=torch.float64)
		dts[0, :] = x[0, :]
		args = [0]
		size = 1
		weights = torch.ones(self.ms)

		for j in range(N):
			point = x[j, :]
			# print (size,x.size())
			if size < self.ms:
				GP.fit_gp(dts[0:size, :], y[0:size, :])
				mean, leverage_score = GP.mean_std(point.view(1, d))
				q = np.random.binomial(self.ms, float(leverage_score))
				if q > 0:
					args.append(j)
					dts[size, :] = point
					weights[size] = (q / float(self.ms)) / leverage_score
					size = size + 1
				else:
					pass
		return (args, weights)

	def fit_gp(self, x, y, eps=1e-14):
		'''
			Function to Fit GP
		'''
		self.x = x
		self.y = y
		self.d = x.size()[1]
		self.N = x.size()[0]
		assert (self.ms <= self.N)
		self.linear_kernel = KernelFunction(kernel_name="linear").linear_kernel
		if self.approx == "svd":
			self.xs = x
			K = self.kernel(x, x)
			if 3 * self.ms > self.N:
				(D, V) = torch.linalg.eigh(K, UPLO='U')
				V = torch.t(V)[self.N - self.ms:self.N, :].T
				D = D[self.N - self.ms:self.N]
				D[D <= eps] = 0

			else:
				(D, V) = torch.lobpcg(K, k=self.ms, niter=-1)

			# Dinv = torch.diag(1./D[self.N-self.ms:self.N])
			# Dinv[Dinv <=0 ] = 0
			# Dinv = torch.sqrt(Dinv)
			self.eigs = D
			Dinv = torch.diag(torch.sqrt(1. / D))
			# self.M = (torch.t(V)[self.N-self.ms:self.N,:]).T @ Dinv.T
			self.M = V @ Dinv
			# self.embed = lambda q: torch.t(torch.mm(Dinv, torch.mm(torch.t(V)[self.N-self.ms:self.N,:], self.kernel(q, self.x)   )))
			self.embed = lambda q: self.kernel(q, self.xs).T @ self.M
			self.C = []
		elif self.approx == 'nothing':
			self.xs = self.x[0:self.ms, :]
			self.M = torch.eye(self.ms).double()
			self.embed = lambda q: self.kernel(q, self.xs).T @ self.M

		elif self.approx == 'positive_svd':
			from sklearn.decomposition import NMF
			GP = GaussianProcess(kernel=self.kernel_object)
			ysample = GP.sample(x, size=self.samples) ** 2
			X = ysample
			model = NMF(n_components=self.ms, max_iter=8000, tol=1e-12)
			W = torch.from_numpy(model.fit_transform(X))
			H = torch.from_numpy(model.components_)
			l = torch.norm(W, dim=1)
			l = 1. / l

			if x.size()[1] == 1:
				fs = []
				for j in range(self.ms):
					fs.append(interp1d(x.view(-1).numpy(), (W.T @ torch.diag(l))[j, :].numpy()))
				self.embed = lambda q: torch.cat([torch.from_numpy(fs[j](q)).view(-1, 1) for j in range(self.ms)],
												 dim=1)

			elif x.size()[1] == 2:
				fs = []
				for j in range(self.ms):
					W_j = (W.T @ torch.diag(l))[j, :].numpy()
					fs.append(LinearNDInterpolator(x, W_j))
				self.embed = lambda q: torch.cat(
					[torch.from_numpy(fs[j](q[:, 0], q[:, 1])).view(-1, 1) for j in range(self.ms)], dim=1)
			# elif x.size()[1] == 2:
			# 	fs = []
			# 	for j in range(self.ms):
			# 		W_j = (W.T @ torch.diag(l))[j, :].numpy()
			# 		fs.append(Rbf(x[:,0],x[:,1], W_j))
			# 	self.embed = lambda q: torch.cat([torch.from_numpy(fs[j](q[:,0],q[:,1])).view(-1, 1) for j in range(self.ms)],
			# 									 dim=1)

			self.C = []

		elif self.approx == "cover":
			K = self.kernel(x, x)  # + self.s * self.s * torch.eye(self.N, dtype=torch.float64)
			Khalf = torch.from_numpy(np.real(scipy.linalg.sqrtm(K.numpy())))
			Khalfinv = torch.pinverse(Khalf)
			self.embed = lambda q: torch.t(
				torch.mm(Khalfinv, self.kernel(q, self.x)))
		else:
			self.C, self.weights = self.subsample(x, y)
			xs = x[self.C, :]
			self.Dweights = torch.diag(self.weights).double()
			K = torch.mm(torch.mm(self.Dweights, self.kernel(xs, xs)),
						 self.Dweights)  # + self.s*self.s * torch.eye(self.ms, dtype=torch.float64)
			#(D, V) = torch.symeig(K, eigenvectors=True)
			(D, V) = torch.linalg.eigh(K)
			Dinv = torch.diag(1. / D)
			Dinv[Dinv <= 0] = 0
			Dinv = torch.sqrt(Dinv)
			# Dinv = torch.diag(torch.pow(D[:],-0.5))
			self.embed = lambda q: torch.t(
				torch.mm(Dinv, torch.mm(torch.t(V), torch.mm(self.Dweights, self.kernel(q, xs)))))
		# self.embed = lambda x: torch.t(torch.mm(torch.sqrt(Dinv),torch.mm(V, self.kernel(x, xs))))
		embeding = self.embed(x)
		self.Z_ = embeding.T @ embeding + self.s * self.s * torch.eye(self.ms).double()

		# self.K = (self.Z_ + self.s * self.s * torch.eye(self.ms, dtype=torch.float64))
		self.K = self.Z_
		self.Q = torch.t(embeding)

		self.fit = True
		return None

	def mean_std(self, xtest):
		if self.fit == False:
			raise AssertionError("First fit")
		else:
			embeding = self.embed(xtest)
			Q = self.embed(self.x)
			theta_mean, _ = torch.solve(torch.mm(torch.t(Q), self.y), self.K)
			ymean = torch.mm(embeding, theta_mean)
			temp = torch.t(torch.solve(torch.t(embeding), self.K)[0])
			diagonal = self.s * self.s * torch.einsum('ij,ji->i', (temp, torch.t(embeding))).view(-1, 1)
			yvar = torch.sqrt(diagonal)

		return (ymean, yvar)

	def outer_kernel(self):
		embeding = self.embed(self.x)
		# print (embeding.size())
		K = torch.mm(embeding, torch.t(embeding))
		# Z = self.linear_kernel(embeding, (embeding))
		K = (K + self.s * self.s * torch.eye(self.N, dtype=torch.float64))
		# K = self.kernel(self.x,self.x) + self.s*self.s*torch.eye(self.N, dtype=torch.float64)
		# print ("kernel:",K)
		# print ("approximate:",Z)
		return K

	def sample_theta(self, size=1):
		basis = int(int(torch.sum(self.m)))
		zeros = torch.zeros(basis, size, dtype=torch.float64)
		random_vector = torch.normal(mean=zeros, std=1.)

		if self.fit == True:
			# random vector
			Z = torch.pinverse(self.K, rcond=10e-6)
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
		f = torch.mm(self.embed(xtest), theta)
		return f

	def visualize(self, xtest, f_true=None, points=True, show=True):
		[mu, std] = self.mean_std(xtest)
		if self.d == 1:

			plt.figure(figsize=(15, 7))
			plt.clf()
			plt.plot(self.x.numpy(), self.y.numpy(), 'r+', ms=10, marker="o")
			plt.plot(self.x[self.C, :].numpy(), self.y[self.C, :].numpy(), 'g+', ms=10, marker="o")
			# plt.plot(xtest.numpy(), self.sample(xtest, size=2).numpy(), 'k--', lw=2, label="sample")
			plt.fill_between(xtest.numpy().flat, (mu - 2 * std).numpy().flat, (mu + 2 * std).numpy().flat,
							 color="#dddddd")
			if f_true is not None:
				plt.plot(xtest.numpy(), f_true(xtest).numpy(), 'b-', lw=2)
			plt.plot(xtest.numpy(), mu.numpy(), 'r-', lw=2, label="posterior mean")
			plt.title('Posterior mean prediction plus 2 st.deviation')
			plt.legend()
			if show == True:
				plt.show()

		elif self.d == 2:
			from scipy.interpolate import griddata
			plt.figure(figsize=(15, 7))
			plt.clf()
			ax = plt.axes(projection='3d')
			xx = xtest[:, 0].numpy()
			yy = xtest[:, 1].numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].numpy(), (grid_x, grid_y), method='linear')
			if f_true is not None:
				grid_z = griddata((xx, yy), f_true(xtest)[:, 0].numpy(), (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z, color='b', alpha=0.4)
			if points == True:
				ax.scatter(self.x[:, 0].numpy(), self.x[:, 1].numpy(), self.y[:, 0].numpy(), c='r', s=100, marker="o",
						   depthshade=False)
			ax.plot_surface(grid_x, grid_y, grid_z_mu, color='r', alpha=0.4)
			plt.title('Posterior mean prediction plus 2 st.deviation')
			plt.show()

		else:
			print("Visualization not implemented")


if __name__ == "__main__":
	# domain size
	L_infinity_ball = 1
	# dimension
	d = 1
	# error variance
	s = 0.1
	# grid density
	n = 1024
	# number of intial points
	N = 100
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

	kernel = KernelFunction(gamma=0.05)
	GP0 = GaussianProcess(kernel_custom=kernel, s=s)
	GP0.fit_gp(x, y)
	GP0.visualize(xtest, f_true=f_no_noise)

	GP = NystromFeatures(kernel, m=torch.Tensor([30]), s=s, approx="uniform")
	GP.fit_gp(x, y)
	GP.visualize(xtest, f_true=f_no_noise)

	GP = NystromFeatures(kernel, m=torch.Tensor([30]), s=s, approx="online_leverage")
	GP.fit_gp(x, y)
	GP.visualize(xtest, f_true=f_no_noise)

	GP = NystromFeatures(kernel, m=torch.Tensor([30]), s=s, approx="svd")
	GP.fit_gp(x, y)
	print(GP0.K, GP.outer_kernel())
	GP.visualize(xtest, f_true=f_no_noise)
