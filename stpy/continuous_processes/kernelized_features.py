from scipy.optimize import minimize
from stpy.helpers.helper import *
from stpy.embeddings.embedding import *
from stpy.kernels import KernelFunction
from stpy.embeddings.transformations import Transformation
from torch.autograd import grad
from stpy.continuous_processes.gauss_procc import GaussianProcess

import torch


class KernelizedFeatures(GaussianProcess):
	'''
		Random Fourier Features for Gaussian Kernel
	'''

	def __init__(self, embedding,m, s=0.001, lam = 1., d=1, diameter=1.0, verbose=True, groups=None,
				 bounds=None, scale=1.0, kappa =1.0, poly = 2, primal = True):

		self.s = s
		self.lam = lam
		self.primal = primal
		self.x = None

		self.K = 0
		self.mu = 0.0

		self.m = torch.from_numpy(np.array(m))
		self.fit = False
		self.data = False

		self.d = d
		self.n = 0
		self.bounds = bounds
		self.groups = groups
		self.diameter = diameter

		self.verbose = verbose
		self.admits_first_order = True

		self.embedding = embedding
		self.embedding_map= embedding

		self.kappa = kappa
		self.scale = scale
		self.poly = poly

		self.to_add = []
		self.prior_mean = 0
		self.linear_kernel = KernelFunction(kernel_name="linear").linear_kernel
		self.dual = False
	def beta(self, delta = 1e-2, norm = 4):

		#beta_value = norm*self.lam + np.sqrt( torch.logdet(self.K/(self.s**2*self.lam)) + 2*np.log(1/delta))
		beta_value = 2.
		return beta_value

	def description(self):
		return "Custom Features object"

	def embed(self, x):
		return self.embedding.embed(x)

	def set_embedding(self,embed):
		self.embedding_map = embed

	def get_basis_size(self):
		return int(torch.sum(self.m))

	def set_basis_size(self,m):
		self.m = m

	def kernel(self,x,y):
		embedding = self.embed(x)
		embedding2 = self.embed(y)
		K = self.linear_kernel(embedding, embedding2)
		return K

	def logdet_ratio(self):
		I = torch.eye(int(torch.sum(self.m))).double()
		return torch.logdet(self.K) - torch.logdet(self.s**2*self.lam*I)

	def effective_dim(self,xtest):
		Phi = self.embed(xtest)
		d = torch.trace(torch.solve(Phi.T@Phi,Phi.T@Phi + torch.eye(self.get_basis_size()).double()*self.lam)[0])
		return d

	def add_data_point(self, x, y):
		if self.n == 0:
			self.fit_gp(x,y)
		else:
			self.to_add.append([x,y])
			self.fit = False

	def fit_gp(self, x, y):
		'''
			Function to Fit GP
		'''
		self.x = x
		self.y = y
		self.n = list(self.x.size())[0]
		self.d = list(self.x.size())[1]

		if self.n < self.m:
			self.dual = True
		else:
			self.dual = False

		if self.primal == True:
			self.dual = False

		self.data = True
		self.fit = False
		return None

	def add_points(self,x,y):
		if self.x is not None:
			self.x = torch.cat((self.x,x),dim=0)
			self.y = torch.cat((self.y,y),dim=0)
		else:
			self.x = x
			self.y = y


	def check_conversion(self):
		"""
		Convert between dual and primal form
		:return:
		"""
		if self.primal == False:
			if self.n == self.m: # convert do d mode
				print ("Switching mode to primal.")
				self.dual = False

				I = torch.eye(int(self.m)).double()
				Z_ = self.linear_kernel(torch.t(self.Q), torch.t(self.Q))
				self.V = (Z_ + self.s * self.s * self.lam * torch.eye(int(self.m), dtype=torch.float64))
				self.invV, _ = torch.solve(I, self.V)

	def get_invV(self):
		self.precompute()

		if self.dual:
			I = torch.eye(self.m).double()
			Z_ = self.linear_kernel(torch.t(self.Q), torch.t(self.Q))
			self.V = (Z_ + self.s * self.s * self.lam * torch.eye(self.m, dtype=torch.float64))
			self.invV, _ = torch.solve(I, self.V)
			return self.invV
		else:
			return self.invV

	def precompute(self):
		if self.fit == False:
			if len(self.to_add)>0:
				# something to add via low rank update
				for i in range(len(self.to_add)):
					newx = self.to_add[i][0]
					newy = self.to_add[i][1]

					# rank one update
					emb = self.embed(newx)

					if self.dual: # via Shur complements
						newKinv = torch.zeros(size = (self.n+1, self.n+1)).double()
						M = self.invK @ self.Q
						c = 1./( (self.s**2 * self.lam + emb@emb.T) - emb@ self.Q.T @ M@emb.T)

						newKinv[0:self.n,0:self.n] = self.invK + c*M @ emb.T @ emb @M.T
						newKinv[0:self.n,self.n] = (- M  @ emb.T * c).view(-1)
						newKinv[self.n,0:self.n] = (- emb @ M.T *c).view(-1)
						newKinv[self.n, self.n] = c.view(-1)

						self.invK = newKinv

						self.add_points(newx, newy)
						self.n = self.n + 1
						self.Q = self.embed(self.x)

						self.invK_V = (1. / self.lam) * (-self.Q.T @ self.invK @ self.Q + torch.eye(int(self.m)))

					else: # via Woodbury
						c = 1 + emb @ self.invV @ emb.T
						self.invV = self.invV - (self.invV @ emb.T @ emb @ self.invV)/c
						self.add_points(newx, newy)
						self.n = self.n + 1
						self.Q = self.embed(self.x)
					# add point

					self.check_conversion()

				self.fit = True
				self.to_add = []


			elif self.data == True: # just compute the
				self.Q = self.embed(self.x)
				if not self.dual:
					I = torch.eye(int(self.m)).double()
					Z_ = self.Q.T@self.Q
					self.V = Z_ + self.s **2 * self.lam *I
					self.invV = torch.pinverse(self.V)
				else:
					I = torch.eye(self.n).double()
					Z_ = self.Q@self.Q.T
					self.K = Z_ + self.s * self.s * self.lam * I
					#self.invK, _ = torch.solve(I, self.K)
					self.invK = torch.pinverse(self.K)
					self.invK_V = (1. / self.lam) * (-self.Q.T @ self.invK @ self.Q + torch.eye(int(self.m)))
				self.fit = True
			else:
				pass
		else:
			pass

	def theta_mean(self, var = False, prior = False):

		self.precompute()
		if self.fit == True and prior == False:
			if self.dual:
				theta_mean = self.Q.T@self.invK@self.y
				Z = self.invK_V
			else:
				theta_mean = self.invV@self.Q.T@self.y
				Z = self.s**2 * self.invV
		else:
			theta_mean = 0*torch.ones(size = (self.m,1)).double()

		if var is False:
			return theta_mean
		else:
			return (theta_mean, Z)

	def mean_std(self, xtest):
		'''
			Calculate mean and variance for GP at xtest points
		'''
		#self.precompute()
		embeding = self.embed(xtest)


		#mean
		theta_mean = self.theta_mean()
		ymean = embeding @theta_mean

		#std
		if not self.dual:
			diagonal = self.s**2*torch.einsum('ij,jk,ik->i', (embeding,self.invV,embeding )).view(-1, 1)
		else:
			diagonal = torch.einsum('ij,jk,ik->i',(embeding,self.invK_V,embeding)).view(-1,1)

		ystd = torch.sqrt(diagonal)
		return (ymean, ystd)

	def sample_matheron(self, xtest, kernel_object, size = 1):
		basis = self.get_basis_size()
		zeros =  torch.zeros(size = (basis , size), dtype=torch.float64)
		random_vector = torch.normal(mean=zeros, std=1.)

		Z = self.lam * torch.eye(basis, dtype=torch.float64)
		L = torch.linalg.cholesky(Z.transpose(-2, -1).conj()).transpose(-2, -1).conj()
		theta = torch.mm(L, random_vector) + self.prior_mean

		f_prior_xtest = torch.mm(self.embed(xtest),theta)
		f_prior_x = torch.mm(self.embed(self.x),theta)

		K_star = kernel_object.kernel(self.x, xtest)
		N = self.x.size()[0]
		K = kernel_object.kernel(self.x,self.x) + self.s**2*self.lam*torch.eye(N)

		f = f_prior_xtest + K_star @ torch.pinverse(K)@(self.y - f_prior_x)
		return f


	def sample_theta(self, size=1, prior = False):

		basis = self.get_basis_size()

		zeros = torch.zeros(size = (basis , size), dtype=torch.float64)
		random_vector = torch.normal(mean=zeros, std=1.)
		self.precompute()

		if self.fit == True and prior == False:
			self.L = torch.linalg.cholesky(self.get_invV())*self.s
			theta = self.theta_mean()
			theta = theta + torch.mm(self.L, random_vector)
		else:
			Z =  self.lam * torch.eye(basis, dtype=torch.float64)
			L = torch.linalg.cholesky(Z.transpose(-2, -1).conj()).transpose(-2, -1).conj()
			theta = torch.mm(L, random_vector) + self.prior_mean

		return theta




	def theta_mean_constrained(self,weights = None, B = 1):
		if weights is None:
			weights = torch.ones(self.n).double()/self.n

		Q = self.embed(self.x)
		theta = cp.Variable(int(torch.sum(self.m).detach().view(-1).numpy()))
		objective = cp.Minimize(cp.sum(weights @ cp.square(Q.detach().numpy() @ theta - self.y.view(-1).detach().numpy())) )
		zero = np.zeros(int(torch.sum(self.m)))
		constraints = [cp.SOC( theta@zero +  B, theta)]
		prob = cp.Problem(objective, constraints)
		prob.solve(solver = cp.MOSEK)
		return torch.from_numpy(theta.value).view(-1,1)


	def theta_absolute_deviation(self, weights = None, reg = None):
		if weights is None:
			weights = torch.ones(self.x.size()[0])
		
		if reg is None: # standard regularization
			Q = self.embed(self.x)
			theta = cp.Variable((int(torch.sum(self.m)),1))
			objective = cp.Minimize(cp.sum(weights@cp.abs(Q.numpy()@theta - self.y.numpy()))+ self.s*self.lam * cp.norm2(theta))
			prob = cp.Problem(objective)
			prob.solve()
			return torch.from_numpy(theta.value)
		else: # custom regularization
			Q = self.embed(self.x)
			theta = cp.Variable((int(torch.sum(self.m)),1))
			objective = cp.Minimize(cp.sum(weights@cp.abs(Q.numpy()@theta - self.y.numpy()))+ reg * cp.norm2(theta))
			prob = cp.Problem(objective)
			prob.solve(solver=cp.MOSEK)
			return torch.from_numpy(theta.value)

	def theta_absolute_deviation_constrained(self, weights = None, B = 1):
		if weights is None:
			weights = torch.ones(self.x.size()[0])
		Q = self.embed(self.x)
		theta = cp.Variable(int(torch.sum(self.m).detach().view(-1).numpy()))

		objective = cp.Minimize(cp.sum(weights @ cp.abs(Q.detach().numpy() @ theta - self.y.view(-1).detach().numpy())))
		zero = np.zeros(int(torch.sum(self.m)))
		constraints = [cp.SOC( theta@zero +  B, theta)]
		prob = cp.Problem(objective, constraints)
		prob.solve(solver = cp.MOSEK)
		return torch.from_numpy(theta.value).view(-1,1)


	def theta_chebyschev_approximation(self, eps = 1.):
		Q = self.embed(self.x).detach().numpy()
		y = self.y.view(-1).detach().numpy()

		theta = cp.Variable(int(torch.sum(self.m).detach().view(-1).numpy()))
		objective = cp.Minimize(cp.sum_squares(theta))
		constraints = [cp.abs(Q @ theta- y)<=eps]

		prob = cp.Problem(objective, constraints)
		prob.solve(solver = cp.MOSEK)
		res = torch.from_numpy(theta.value).view(-1,1)
		return res

	def interpolation(self, eps = 0.):
		Q = self.embed(self.x).detach().numpy()
		y = self.y.view(-1).detach().numpy()
		theta = cp.Variable(int(torch.sum(self.m).detach().view(-1).numpy()))
		objective = cp.Minimize(cp.sum_squares(theta))
		constraints = [Q @ theta == y]

		prob = cp.Problem(objective, constraints)
		prob.solve()
		res = torch.from_numpy(theta.value).view(-1,1)

		return res



	def mean_squared(self, xtest, weights = None, B = None, theta = False, reg = None):
		embeding = self.embed(xtest)

		if B is not None:
			theta_mean = self.theta_mean_constrained(weights = weights, B = B)
		else:
			theta_mean = self.theta_mean(weights = weights, reg = reg)
		ymean = torch.mm(embeding, theta_mean)
		if theta == True:
			return ymean, theta_mean
		else:
			return ymean

	def mean_aboslute_deviation(self, xtest, weights = None, B = None, theta = False):
		embeding = self.embed(xtest)
		if B is not None:
			theta_mean = self.theta_absolute_deviation_constrained(weights=weights, B=B)
		else:
			theta_mean = self.theta_absolute_deviation(weights=weights)
		ymean = torch.mm(embeding, theta_mean)
		if theta == True:
			return ymean, theta_mean
		else:
			return ymean



	"""
	Hessian 
	"""

	def mean_gradient_hessian(self,xtest,hessian = False):
		hessian_mu = torch.zeros(size=(self.d,self.d), dtype = torch.float64)
		xtest.requires_grad_(True)
		#xtest.retain_grad()
		mu = self.mean_std(xtest)[0]
		#mu.backward(retain_graph=True)

		#nabla_mu = xtest.grad
		nabla_mu = grad(mu, xtest, create_graph=True)[0][0]

		if hessian == False:
			return nabla_mu
		else:
			for i in range(self.d):
				hessian_mu[i, :]= grad(nabla_mu[i], xtest, create_graph=True, retain_graph=True)[0][0]
			return [nabla_mu, hessian_mu]


	""" 
	Optimization
	"""

	def ucb_optimize(self,beta, multistart = 25, lcb = False, minimizer = "L-BFGS-B"):

		# precompute important (theta)
		theta_mean, K = self.theta_mean(var = True)

		if lcb == False:
			fun = lambda x: - (self.embed(torch.from_numpy(x).view(1,-1))@theta_mean +\
							  beta*torch.sqrt(self.embed(torch.from_numpy(x).view(1,-1))@K@self.embed(torch.from_numpy(x).view(1,-1)).T )).detach().numpy()[0]
		else:
			fun = lambda x:- (self.embed(torch.from_numpy(x).view(1,-1))@theta_mean - \
							beta*torch.sqrt(self.embed(torch.from_numpy(x).view(1,-1))@K@self.embed(torch.from_numpy(x).view(1,-1)).T).detach().numpy()[0]).numpy()[0]


		if self.bounds == None:
			mybounds = tuple([(-self.diameter, self.diameter) for _ in range(self.d)])
		else:
			mybounds = self.bounds

		results = []
		for j in range(multistart):

			x0 = np.random.randn(self.d)
			for i in range(self.d):
				x0[i] = np.random.uniform(mybounds[i][0], mybounds[i][1])

			if minimizer== "L-BFGS-B":
				res = minimize(fun, x0, method="L-BFGS-B", jac=None, tol=0.0001, bounds=mybounds)
				solution = res.x
			else:
				raise AssertionError("Wrong optimizer selected.")


			results.append([solution, -fun(solution)])

		results = np.array(results)
		index = np.argmax(results[:, 1])
		solution = results[index, 0]
		return (torch.from_numpy(solution).view(1,-1), -torch.from_numpy(fun(solution)))


	def sample_and_optimize(self, xtest = None, multistart=25, minimizer = "L-BFGS-B", grid = 100, verbose = 0):
		'''
			Sample functions from Gaussian Process and take Maximum using
			first order maximization
		'''

		# sample linear approximating
		theta = self.sample_theta()

		# get bounds
		if self.bounds == None:
			mybounds = tuple([(-self.diameter, self.diameter) for _ in range(self.d)])
		else:
			mybounds = self.bounds

		fun = lambda x: -torch.mm(torch.t(theta),torch.t(self.embed(torch.from_numpy(x).view(1,-1)))).numpy()[0]

		results = []
		for j in range(multistart):
			x0 = np.random.randn(self.d)
			for i in range(self.d):
				x0[i] = np.random.uniform(mybounds[i][0], mybounds[i][1])

			if minimizer== "L-BFGS-B":
				res = minimize(fun, x0, method="L-BFGS-B", jac=None, tol=0.0001, bounds=mybounds)
				solution = res.x
			else:
				raise AssertionError("Wrong optimizer selected.")


			results.append([solution, -fun(solution)])
		results = np.array(results)
		index = np.argmax(results[:, 1])
		solution = results[index, 0]

		return (torch.from_numpy(solution), -torch.from_numpy(fun(solution)))


	def sample(self, xtest, size=1, prior = False):
		'''
			Sample functions from Gaussian Process
		'''
		theta = self.sample_theta(size=size, prior = prior)
		f = torch.mm(self.embed(xtest),theta)
		return f


	def sample_and_max(self, xtest, size=1):
		'''
			Sample functions from Gaussian Process and take Maximum
		'''
		f = self.sample(xtest, size=size)
		index = np.argmax(f, axis=0)
		return (xtest[index, :], f[index, :])

	def get_kernel(self):
		embeding = self.embed(self.x)
		Z_ = self.linear_kernel(embeding, embeding)
		K = (Z_ + self.s * self.s * self.lam * torch.eye(int(self.n), dtype=torch.float64))
		return K

	def residuals(self):
		mu, _ = self.mean_std(self.x)
		out = torch.sum((mu - self.y)**2)
		return out


if __name__ == "__main__":

	N = 10
	s = 0.1
	n = 256
	L_infinity_ball = 0.5

	d = 1
	m = 128

	xtest = torch.from_numpy(interval(n,d,L_infinity_ball=L_infinity_ball))
	x = torch.from_numpy(np.random.uniform(-L_infinity_ball,L_infinity_ball,N)).view(-1,1)

	F_true = lambda x: torch.sin(x*4)**2-0.1
	F = lambda x: F_true(x) + s*torch.randn(x.size()[0]).view(-1,1).double()
	y = F(x)

	emb = RFFEmbedding(m=m, gamma = 0.1)
	Reggr = KernelizedFeatures(embedding=emb, m = m, d = 1)
	Reggr.fit_gp(x,y)
	Reggr.visualize(xtest, f_true= F_true)

