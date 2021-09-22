import numpy as np
import torch
import scipy
import mosek
import cvxpy as cp
from stpy.helpers.quadrature_helper import quadvec2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from stpy.embeddings.embedding import HermiteEmbedding
import scipy.integrate as integrate
from stpy.helpers.ellipsoid_algorithms import maximize_quadratic_on_ellipse, minimize_quadratic_on_ellipse
from stpy.helpers.ellipsoid_algorithms import maximize_matrix_quadratic_on_ellipse, minimize_matrix_quadratic_on_ellipse
from stpy.point_processes.poisson import PoissonPointProcess
from stpy.point_processes.poisson_rate_estimator import PositiveRateEstimator
from stpy.borel_set import BorelSet, HierarchicalBorelSets
from stpy.kernels import KernelFunction

## implement loading data

class PermanentalProcessRateEstimator(PositiveRateEstimator):

	def __init__(self, *args, **kwargs):
		super().__init__(*args,**kwargs)

		self.integration = "fixed_quad"
		self.product_integrals = {}
		self.varLambdas = torch.zeros(size=(len(self.basic_sets), self.get_m(),self.get_m())).double()
		self.opt = 'cvxpy'
		if self.feedback == "count-record" and self.estimator=="least-sq":
			print ("precomputing-integrals:")
			for index_set, set in enumerate(self.basic_sets):
				print (index_set,"/",len(self.basic_sets))
				self.varLambdas[index_set, :] = self.product_integral(set)
				self.variances[index_set] = set.volume() * self.B


	def product_integral(self,S):

		if S in self.product_integrals.keys():
			return self.product_integrals[S]
		else:

			if "product_integral" in dir(self.packing):
				Psi = self.packing.product_integral(S)
				self.product_integrals[S] = Psi
				return Psi

			elif self.integration ==  "vec_quad":

				if S.d == 2:
					#Psi = torch.zeros(size=(self.get_m(), self.get_m())).double()
					F = lambda x: (self.packing.embed(x).view(-1, 1) @\
								   self.packing.embed(x).view(1, -1)).view(-1)
					integrand = lambda x, y: F(torch.Tensor([x, y]).view(1, 2).double()).numpy()

					val = quadvec2(integrand,float(S.bounds[0, 0]), float(S.bounds[0, 1]),
								   float(S.bounds[1, 0]), float(S.bounds[1, 1]),limit = 10,epsrel = 10e-3, epsabs = 10e-3, quadrature = 'gk15')
					Psi = torch.from_numpy(val).view((self.get_m(), self.get_m()))

			elif self.integration == "fixed_quad":

				if S.d ==1:
					weights, nodes = S.return_legendre_discretization(n=128)
					Z = self.packing.embed(nodes)
					M = torch.einsum('ij,ik->ijk', Z, Z)
					Psi = torch.einsum('i,ijk->jk', weights, M)

				if S.d ==2:
					weights, nodes = S.return_legendre_discretization(n = 50)
					Z = self.packing.embed(nodes)
					M = torch.einsum('ij,ik->ijk',Z,Z)
					Psi = torch.einsum('i,ijk->jk',weights,M)

			else:
				Psi = torch.zeros(size = (self.get_m(),self.get_m())).double()
				for i in range(self.get_m()):
					for j in range(self.get_m()):

						if S.d == 1:
							F_ij = lambda x: (self.packing.embed(torch.from_numpy(np.array(x)).view(1, -1)).view(-1)[i] *
											  self.packing.embed(torch.from_numpy(np.array(x)).view(1, -1)).view(-1)[
												  j]).numpy()
							val, status = integrate.quad(F_ij,float(S.bounds[0,0]), float(S.bounds[0,1]))


						elif S.d == 2:
							F_ij = lambda x:  self.packing.embed(x).view(-1)[i] *self.packing.embed(x).view(-1)[j]
							integrand = lambda x, y: F_ij(torch.Tensor([x, y]).view(1, 2).double()).numpy()
							val,status = integrate.dblquad(integrand, float(S.bounds[0, 0]), float(S.bounds[0, 1]),
															lambda x: float(S.bounds[1, 0]),
															lambda x: float(S.bounds[1, 1]),epsabs=1.49e-03, epsrel=1.49e-03)
						else:
							raise NotImplementedError("Integration above d>2 not implemented.")

						Psi[i,j] = val
						print(i, j, val)

			self.product_integrals[S] = Psi
			return Psi

	def get_constraints(self):
		s = self.get_m()
		l = np.full(s, self.b)
		u = np.full(s, self.B)
		Lambda = np.identity(s)
		return (l,Lambda,u)

	def cov(self, inverse=False):
		s = self.get_m()

		if inverse==False:
			return torch.zeros(size = (s,s)).double()
		else:
			return torch.zeros(size=(s, s)).double(),torch.zeros(size=(s, s)).double()


	def sample(self, verbose = False, steps = 10, stepsize = None):

		if self.data is None:
			self.sampled_theta = torch.zeros(self.get_m()).double().view(-1,1)
			return None

		if self.observations is not None:
			observations = self.observations.double()
			sumLambda = self.sumLambda.double()
			nabla = lambda theta: -torch.sum(torch.diag(1. /(observations@theta).view(-1)) @ observations)	\
								  + (sumLambda.T + sumLambda) @ theta + self.s*theta.view(-1,1)
		else:
			sumLambda = self.sumLambda.double()
			nabla = lambda theta: (sumLambda.T + sumLambda) @ theta + self.s*theta.view(-1,1)

		theta = self.rate.view(-1, 1)

		W = self.construct_covariance_matrix_laplace()
		L = float(scipy.sparse.linalg.eigsh(W.numpy(), k=1, which='LM', return_eigenvectors=False, tol=1e-3))
		eta = 0.5 / (L + 1)

		for k in range(steps):
			W = torch.randn(size=(self.get_m(), 1)).double()
			theta = theta - eta * nabla(theta) + np.sqrt(2 * eta) * W
			if verbose == True:
				print("Iter:", k, theta.T)

		self.sampled_theta = theta
		return None

	def sample_value(self, S):
		"""
		Given a pre-sampled value evaluate certain portions of the domain S
		:param S:
		:return:
		"""
		Z = self.product_integral(S)
		map = self.sampled_theta.T@ Z @self.sampled_theta
		return map


	def sample_path(self, S, n=128):
		xtest = S.return_discretization(n)
		return (self.packing.embed(xtest) @ self.sampled_theta)**2




	def load_data(self,data):
		super().load_data(data, times = False)
		self.sumLambda = torch.zeros(size = (self.get_m(),self.get_m()))
		if len(data) > 1:
			for sample in data:
				(S,obs,dt) = sample
				self.sumLambda += self.product_integral(S) * dt

	def add_data_point(self, new_data):
		super().add_data_point(new_data, times = False)
		(S, obs, dt) = new_data
		self.sumLambda += self.product_integral(S) * dt

	def penalized_likelihood(self, threads = 4):
		sumLambda = self.sumLambda.numpy()
		if self.observations is not None:
			observations = self.observations.numpy()
			loss = lambda theta: float(-np.sum(np.log(  (observations@theta)**2 ))  + np.dot(theta, sumLambda@theta) + 0.5*self.s*np.sum(theta**2))
		else:
			loss = lambda theta: float(np.dot(theta, sumLambda @ theta) + 0.5*self.s * np.sum(theta ** 2))

		theta = np.random.randn(self.get_m())
		res = minimize(loss, theta, jac=None, method='L-BFGS-B')
		self.rate = torch.from_numpy(res.x)
		return self.rate

	def construct_covariance_matrix_laplace(self):
		W = torch.zeros(size=(self.get_m(), self.get_m())).double()

		if self.feedback == "count-record":
			if self.observations is not None:
				for i in range(self.observations.size()[0]):
					A = self.observations[i, :].view(-1, 1) @ self.observations[i, :].view(1, -1)
					k = np.maximum(torch.dot(self.observations[i, :],self.rate.view(-1)) ** 2,self.b)
					W = W + A / k
			W += 2*self.sumLambda
		else:
			raise AssertionError("Not implemented.")
		return W + torch.eye(self.get_m()).double()*self.s


	def map_lcb_ucb_approx_action(self, S, dt=1.,  beta=2.):

		phi = self.packing.integral(S)
		map = (phi @ self.rate)

		ucb = np.maximum((map + beta*np.sqrt(phi@self.W_inv_approx@phi.T))**2,(map - beta*np.sqrt(phi@self.W_inv_approx@phi.T))**2)
		ucb = np.minimum(ucb,self.B*S.volume()*dt)
		lcb = 0.

		return dt*map**2, dt*lcb, dt*ucb

	def mean_std_per_action(self,S,W, dt , beta):
		Z = self.product_integral(S)

		ucb, _ = maximize_matrix_quadratic_on_ellipse(Z.numpy(), (W).numpy(), self.rate.view(-1).numpy(), beta)
		lcb, _ = minimize_matrix_quadratic_on_ellipse(Z.numpy(), (W).numpy(), self.rate.view(-1).numpy(), beta)

		map = self.rate.T @ Z @ self.rate

		return dt * map, dt * ucb, -lcb * dt


	def mean_rate(self, S, n=128):
		xtest = S.return_discretization(n)
		return (self.packing.embed(xtest) @ self.rate)**2

	def mean_rate_latent(self,S,n = 128):
		xtest = S.return_discretization(n)
		return self.packing.embed(xtest) @ self.rate


	def map_lcb_ucb_approx(self,S,n,beta = 2.0, delta = 0.01):
		xtest = S.return_discretization(n)
		if self.data is None:
			return  0 * xtest[:, 0].view(-1, 1),self.b + 0 * xtest[:, 0].view(-1, 1), self.B + 0 * xtest[:,0].view(-1,xtest.size()[0])
		self.fit_ellipsoid_approx()

		Phi = self.packing.embed(xtest).double()
		map = Phi @ self.rate
		N = Phi.size()[0]

		ucb = torch.zeros(size=(N, 1)).double()
		lcb = torch.zeros(size=(N, 1)).double()

		for i in range(N):
			x = Phi[i, :].view(-1,1)
			maximum = np.maximum((map[i] - beta * np.sqrt(x.T @ self.W_inv_approx @ x))**2, (map[i] + beta * np.sqrt(x.T @ self.W_inv_approx @ x))**2)
			ucb[i,0] = np.minimum( maximum ,self.B)
			lcb[i,0] = 0.
			#lcb[i, 0] = map[i] - np.sqrt(beta) * np.sqrt(x.T @ self.W_inv_approx @ x) ** 2
		return map**2, lcb, ucb

	def map_lcb_ucb(self, S, n, beta = 2.0, delta = 0.01):
		"""
		Calculate exact confidence using laplace approximation on a whole set domain
		:param S: set
		:param n: discretization
		:param beta: beta
		:return:
		"""

		xtest = S.return_discretization(n)
		if self.data is None:
			return self.b+0*xtest[:,0].view(-1,1),self.b+0*xtest[:,0].view(-1,1),self.B+0*xtest[:,0].view(-1,1)

		N = xtest.size()[0]
		Phi = self.packing.embed(xtest)
		map = (Phi @ self.rate)**2

		if self.uncertainty == "laplace":
			W = self.construct_covariance_matrix_laplace()
		ucb = torch.zeros(size=(N, 1)).double()
		lcb = torch.zeros(size=(N, 1)).double()

		for i in range(N):
			x = Phi[i, :]
			ucbi, _ = maximize_quadratic_on_ellipse(x.numpy(), (W).numpy(), self.rate.view(-1).numpy(), beta)
			lcbi, _ = minimize_quadratic_on_ellipse(x.numpy(), (W).numpy(), self.rate.view(-1).numpy(), beta)
			ucb[i, 0] = ucbi
			lcb[i, 0] = lcbi

		return map, lcb, ucb


class LogisticGaussProcessRateEstimator(PermanentalProcessRateEstimator):

	def penalized_likelihood(self, threads=4):
		logistic = lambda x: np.log(1 + np.exp(x))
		weights = self.weights.numpy()
		nodes = self.nodes.numpy()

		if self.observations is not None:
			observations = self.observations.numpy()
			loss = lambda theta: float(-np.sum(np.log(logistic(observations @ theta))) + np.sum(
				weights * logistic(theta @ nodes.T)) + self.s * np.sum(theta ** 2))
		else:
			loss = lambda theta: float(np.sum(weights * logistic(theta @ nodes.T)) + self.s * np.sum(theta ** 2))

		theta = np.random.randn(self.get_m())
		res = minimize(loss, theta, jac= None, method='L-BFGS-B',options={'maxcor': 20,'iprint':-1,'maxfun':150000,'maxls': 50})
		self.rate = torch.from_numpy(res.x)

		return self.rate

	def logistic(self, x):
		return torch.log(1 + torch.exp(x))

	def mean_rate(self, S, n=128):
		xtest = S.return_discretization(n)
		return self.logistic(self.packing.embed(xtest) @ self.rate)


class ExpGaussProcessRateEstimator(PermanentalProcessRateEstimator):


	def penalized_likelihood(self, threads=4):
		weights = self.weights.numpy()
		nodes = self.nodes.numpy()

		if self.observations is not None:
			observations = self.observations.numpy()
			loss = lambda theta: float(np.sum(observations @ theta) + np.sum(
				weights * np.exp(-theta @ nodes.T)) + self.s * np.sum(theta ** 2))
		else:
			loss = lambda theta: float(np.sum(weights * np.exp(-theta @ nodes.T)) + self.s * np.sum(theta ** 2))

		theta = np.zeros(self.get_m())
		res = minimize(loss, theta, jac= None, method='L-BFGS-B',options={'maxcor': 20,'iprint':-1,
																		  'maxfun':150000,'maxls': 100,
																		  'ftol':1e-12,'eps':1e-12,'gtol':1e-8})
		self.rate = torch.from_numpy(res.x)

		return self.rate

	def mean_rate(self, S, n=128):
		xtest = S.return_discretization(n)
		return torch.exp(-self.packing.embed(xtest) @ self.rate)

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
	levels = 4
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	Sets = hierarchical_structure.get_all_sets()

	D = BorelSet(1, bounds=torch.Tensor([[-1., 1.]]).double())

	m = 64
	embedding = HermiteEmbedding(m = m, d = 1, gamma = gamma)
	k = KernelFunction(gamma = gamma)

	estimator5 = PositiveRateEstimator(process, hierarchical_structure, kernel_object=k, B=B, m=m, d = d)

	estimator4 = PermanentalProcessRateEstimator(process, hierarchical_structure,kernel_object=k, B=B, m=m, d = d)
	#estimator = PermanentalProcessRateEstimator(process, hierarchical_structure,
	#											kernel_object=k, B=B, m=m, d=d, embedding=embedding, basis = "custom", approx="ellipsoid")
	#estimator = LogGaussProcessRateEstimator(process, hierarchical_structure, kernel_object=k, B=B, m=m, d=d, embedding=embedding, basis = "custom")
	estimator = LogGaussProcessRateEstimator(process, hierarchical_structure, kernel_object=k, B=B+1, m=m, d=d, embedding=embedding)

	#estimator = LogisticGaussProcessRateEstimator(process, hierarchical_structure, kernel_object=k, B=B, m=m, d=d, embedding=embedding, basis = "custom")
	estimator2 = LogisticGaussProcessRateEstimator(process, hierarchical_structure, kernel_object=k, B=B, m=m, d=d, embedding=embedding)
	#estimator = ExpGaussProcessRateEstimator(process, hierarchical_structure, kernel_object=k, B=B, m=m, d=d, embedding=embedding, basis = "custom")
	estimator3 = ExpGaussProcessRateEstimator(process, hierarchical_structure, kernel_object=k, B=B, m=m, d=d, embedding=embedding)

	estimators = [estimator,estimator2,estimator3,estimator4,estimator5]
	names = ['sigmoid','logistic','exp','square','no-link']
	bands = [True,False,False,False,True]


	estimators = [estimator,estimator5,estimator4]
	names = ['sigmoid','no-link','square']
	bands = [False,False,False]

	min_vol, max_vol = estimator.get_min_max()
	dt = 10. / (b * min_vol)
	dt = dt * 2

	print("Suggested dt:", dt)
	c = ['k', 'r', 'b', 'y', 'g', 'orange', 'brown', 'purple'] + ['k' for i in range(500)]

	no_sets = len(Sets)


	# no_samples = 3
	# data = []
	# samples = []
	# repeats = 2
	#
	# for i in range(no_samples):
	# 	j = np.random.randint(0, no_sets, 1)
	# 	S = Sets[j[0]]
	# 	for _ in range(repeats):
	# 		sample = process.sample_discretized(S, dt)
	# 		samples.append(sample)
	# 		data.append((S, sample, dt))
	#
	# sample_D = process.sample_discretized(D, dt)
	# samples.append(sample_D)
	# no_samples = repeats * no_samples + 1
	# data.append((D, sample_D, dt))


	data_single = []
	basic_sets = hierarchical_structure.get_sets_level(levels)
	samples = []

	for set in basic_sets:
		sample = process.sample_discretized(set,dt)
		data_single.append((set,sample,dt))
		samples.append(sample)
	data = data_single

	# sample_D = torch.cat(samples)
	# data = [(D,sample_D,dt)]

	# data2 = []
	# samples = []
	# for set in basic_sets:
	# 	sample = process.sample_discretized(set,dt*2)
	# 	data2.append((set,sample,dt*2))
	# 	samples.append(sample)
	#
	# sample_D_2 = torch.cat(samples)
	# data = [(D, sample_D_2, dt*2)]
	#
	# data = data + data2

	for estimator,name,band in zip(estimators,names,bands):
		estimator.load_data(data)

		xtest = D.return_discretization(n=n)

		# likelihood based
		estimator.fit_gp()
		rate_mean = estimator.mean_rate(D,n = n)
		p = plt.plot(xtest, rate_mean, label='likelihood: '+name)

		if band == True:
			_, lcb, ucb = estimator.map_lcb_ucb(D, n, beta=2.)
			plt.fill_between(xtest.numpy().flatten(), lcb.numpy().flatten(), ucb.numpy().flatten(), alpha=0.4,
							 color=p[0].get_color(),	 label=name)



	for j in range(len(samples)):
		if samples[j] is not None:
			plt.plot(samples[j], samples[j] * 0, 'o', color=c[j])

	# for action in Sets:
	# 	map, lcb, ucb = estimator.map_lcb_ucb_approx_action(action,beta=2.)
	# 	x = np.linspace(action.bounds[0,0],action.bounds[0,1],2)
	# 	plt.plot(x,x*0+float(ucb/action.volume()),'-o', color = "green")
	process.visualize(D, samples=0, n=n, dt=1.)
	plt.show()
