from abc import abstractmethod, ABC
from typing import Union
import scipy

import torch
import mosek
import numpy as np
import cvxpy as cp


class NoiseModel(ABC):
	"""
	Class provides an interface to sample noise observations and evaluate their likelihood
	"""
	def __init__(self):
		pass

	@abstractmethod
	def sample(self, xs, theta):
		pass

	@abstractmethod
	def sample_noise(self, xs):
		pass

	def joint_log_likelihood(self, ys, xs, theta: Union[np.array, cp.Variable]) -> Union[np.array, cp.Expression]:
		""" Returns the sum of the lls, i.e. the joint ll"""
		if isinstance(theta, cp.Variable):
			return cp.sum(self.log_likelihood(ys, xs, theta))
		else:
			return np.sum(self.log_likelihood(ys, xs, theta))



	def get_mosek_params(self, threads=4):
		if self.convex:
			return {
				mosek.iparam.num_threads: threads,
				mosek.iparam.intpnt_solve_form: mosek.solveform.primal,
				mosek.dparam.intpnt_co_tol_pfeas: 1e-4,
				mosek.dparam.intpnt_co_tol_dfeas: 1e-4,
				mosek.dparam.intpnt_co_tol_rel_gap: 1e-4
			}
		else:
			raise AttributeError("Fetching mosek parameters disallowed for non-convex problems")

	@abstractmethod
	def convex(self) -> bool:
		pass


class AdditiveHomoscedasticNoiseModel(NoiseModel):
	"""
	Assume a linear model. Only thing left to implement is the eta log-likelihood in both cvxpy and numpy

	TODO discuss whether xs @ theta should be replaced by a f_noiseless type function you can pass at initialization?
	"""
	@abstractmethod
	def sample_noise(self, xs):
		""" pass xs in order to know how large noise should be. Also able to deal with heteroscedastic later on """
		pass

	def sample(self, xs, theta):
		return xs @ theta + self.sample_noise(xs)

	def log_likelihood(self, ys, xs, theta):  # TODO change base class
		if ys.shape[0] == 0:
			return 0. # this is to avoid problems with cvxpy variables of size 0, which it doesn't like
		if isinstance(theta, cp.Variable):
			return self.cvxpy_noise_log_likelihood(ys - (xs @ theta))
		else:
			return self.noise_log_likelihood(ys - (xs @ theta))



class PoissonNoise(NoiseModel):

	def __init__(self, lam):
		self.lam = lam

	def sample_noise(self, xs):
		return torch.poisson(self.lam(xs).view(-1)).view(-1,1)
	def convex(self) -> bool:
		pass

	def sample(self, xs, theta):
		pass

	def mean(self, xs):
		return self.lam(xs)
class GaussianNoise(AdditiveHomoscedasticNoiseModel):
	def __init__(self, sigma=0.1):
		"""
		:param sigma: standard deviation
		"""
		super().__init__()
		self.sigma = sigma

	def sample_noise(self, xs):
		return self.sigma*np.random.normal(scale=1.0, size=(xs.shape[0], 1))

	def noise_log_likelihood(self, etas, xs=None):
		return -(0.5*((etas) ** 2))/(self.sigma **  2) - 0.5*np.log(2*np.pi*(self.sigma**2))

	def cvxpy_noise_log_likelihood(self, etas, xs=None):
		return -0.5 * cp.square(etas) / (self.sigma ** 2) - 0.5*np.log(2 * np.pi * self.sigma ** 2)

	@property
	def convex(self) -> bool:
		return True

	def __str__(self):
		return "GaussianAdditive"



class HuberNoise(AdditiveHomoscedasticNoiseModel):
	def __init__(self, sigma=0.1):
		"""
		:param sigma: standard deviation
		"""
		super().__init__()
		self.sigma = sigma

	def sample_noise(self, xs):
		return self.sigma*(np.random.normal(scale=1.0, size=(xs.shape[0], 1)) +  np.random.laplace(scale=self.sigma, size=(xs.shape[0], 1)))/2.

	@property
	def convex(self) -> bool:
		return True

	def __str__(self):
		return "GaussianAdditive"

class AdditiveBoundedNoise(GaussianNoise):
	""" Sub-Gaussian bounded norm, with a Gaussian Likelihood"""
	def __init__(self, lower, upper):
		super().__init__(upper-lower)
		self.lower = lower
		self.upper = upper

	def sample_noise(self, xs):
		raw = np.random.random_sample(size=(xs.shape[0], 1))
		rescaled = self.lower + raw * self.sigma
		print(rescaled)
		return rescaled  # sigma is the length of the interval

	def __str__(self):
		return "BoundedNoiseAdditive"


class MisspecifiedAdditiveGaussianNoise(GaussianNoise):
	def __init__(self, sigma=1.0, actual_sigma=0.1):
		"""
		:param sigma: standard deviation
		"""
		super().__init__(sigma=sigma)
		self.actual_sigma = actual_sigma

	def sample_noise(self, xs):
		return self.actual_sigma*np.random.normal(scale=1.0, size=(xs.shape[0], 1))

	def __str__(self):
		return "MisspecifiedGaussianAdditive"


class LaplaceNoise(GaussianNoise):
	def __init__(self, b):
		"""
		:param sigma: this is sometimes also denoted as b
		"""
		super().__init__()
		self.b = b

	def noise_log_likelihood(self, etas):
		return -np.log(2*self.b) - np.abs(etas)/self.b

	def cvxpy_noise_log_likelihood(self, etas):
		return -np.log(2*self.b) - cp.abs(etas)/self.b

	def sample_noise(self, xs):
		return np.random.laplace(loc = 0, scale=self.b, size=(xs.shape[0], 1))

	def __str__(self):
		return "Laplace"

	@property
	def convex(self) -> bool:
		return True


class AdditiveGumbelNoise(AdditiveHomoscedasticNoiseModel):
	def __init__(self, beta, mu):
		super().__init__()
		self.beta = beta
		self.mu = mu

	def sample_noise(self, xs):
		return np.random.gumbel(loc=self.mu, scale=self.beta, size=(xs.shape[0],))

	def noise_log_likelihood(self, etas):
		return -np.log(self.beta) - 1/self.beta*(etas - self.mu) - np.exp(-1/self.beta*(etas-self.mu))

	def cvxpy_noise_log_likelihood(self, etas):
		return -np.log(self.beta) - 1/self.beta*(etas - self.mu) - cp.exp(-1/self.beta*(etas-self.mu))

	def __str__(self):
		return "Gumbel"

	@property
	def convex(self) -> bool:
		return True

class AdditiveTwoSidedWeibullNoise(AdditiveHomoscedasticNoiseModel):
	def __init__(self, scale, shape):
		"""
		:param scale: lambda
		:param shape: k
		"""
		super().__init__()
		self.scale = scale
		self.shape = shape

	def noise_log_likelihood(self, etas):
		etas = np.abs(etas)
		return np.log(0.5*self.shape/self.scale) + (self.shape - 1)*np.log(etas/self.scale) - np.power(etas/self.scale, self.shape)

	def cvxpy_noise_log_likelihood(self, etas):
		raise NotImplementedError("cvxpy makes no sense for non-convex sets")

	def sample_noise(self, xs):
		signs = np.sign(np.random.normal(size=xs.shape[0]))
		weibull = np.random.weibull(self.shape, size=xs.shape[0])
		return self.scale * signs * weibull

	def __str__(self):
		return "TwoSidedWeibull"

	@property
	def convex(self) -> bool:
		return False

class BernoulliNoise(NoiseModel):

	def __init__(self, prob):
		"""
		:param scale: lambda
			Note lambda should work for both cvxpy and np parameter inputs and takes xs, theta
		:param shape: p
		"""
		super().__init__()
		self.prob = prob # lambda , $lambda^(1/a) to connect to sampling below

	def mean(self, xs):
		return self.prob(xs)

	def sample_noise(self, xs):
		bernouli = torch.bernoulli(self.prob(xs).view(-1))
		return bernouli.view(-1,1)

	def convex(self):
		pass

	def sample(self, xs, theta):
		pass

	def log_likelihood(self, ys, xs, theta: Union[np.array, cp.Variable]) -> Union[np.array, cp.Expression]:
		pass


class LogWeibullNoise(NoiseModel):
	def __init__(self, lam, p = 2, lam_form = lambda x, y: np.exp(x@y)):
		"""
		:param scale: lambda
			Note lambda should work for both cvxpy and np parameter inputs and takes xs, theta
		:param shape: p
		"""
		super().__init__()
		self.lam = lam # lambda , $lambda^(1/a) to connect to sampling below 
		self.p = p #  
		self.lam_form = lam_form

	def sample(self,xs,theta):
		pass

	def log_likelihood(self, ys, xs, theta):
		assert(xs is not None)
		if isinstance(theta, cp.Variable):
			return self.cvxpy_log_likelihood(ys, xs, theta)
		else:
			return self.noise_log_likelihood(ys, xs, theta)
	
	def noise_log_likelihood(self,ys, xs, theta):
		return np.log(self.lam_form(xs, theta).reshape(-1)) + self.p*ys.reshape(-1) - np.exp(ys).reshape(-1)**self.p*self.lam_form(xs, theta).reshape(-1)
		# notice that lam(xs) = exp(\theta^\top xs) in common parametrization hence the loglikelihood becomes 
		# xs @ theta + p*y - np.exp(y)**p*np.exp(xs@\theta) # which is strongly convex in theta

	def sample_noise(self, xs):
		weibull = (self.lam(xs)**(1/self.p)).reshape(-1)*np.random.weibull(self.p, size=xs.shape[0])
		weibull = weibull.reshape(-1,1)
		return np.log(weibull)

	def mean(self, xs):
		return (np.log(self.lam(xs)) - np.euler_gamma)/self.p

	def cvxpy_log_likelihood(self, ys, xs, theta):
		# This works only fi 
		return xs @ theta + self.p*ys - cp.multiply((np.exp(ys)**self.p).reshape(-1),cp.exp(xs@theta))

	def __str__(self):
		return "logWeibull"

	@property
	def convex(self) -> bool:
		return True

class WeibullNoise(LogWeibullNoise):

	def noise_log_likelihood(self,ys, xs, theta):
		return np.log(self.lam_form(xs, theta).reshape(-1)) + np.log(self.p * (ys.reshape(-1)**(self.p-1))) - self.lam_form(xs, theta).reshape(-1)*(ys.reshape(-1)**self.p)
		# notice that lam(xs) = exp(\theta^\top xs) in common parametrization hence the loglikelihood becomes
		# xs @ theta + p*y - np.exp(y)**p*np.exp(xs@\theta) # which is strongly convex in theta

	def noise_likelihood(self,ys, xs, theta):
		return self.lam_form(xs, theta).reshape(-1)*(self.p * (ys.reshape(-1)**(self.p-1)))*np.exp(- self.lam_form(xs, theta).reshape(-1)*(ys.reshape(-1)**self.p))
		# notice that lam(xs) = exp(\theta^\top xs) in common parametrization hence the loglikelihood becomes
		# xs @ theta + p*y - np.exp(y)**p*np.exp(xs@\theta) # which is strongly convex in theta

	def sample_noise(self, xs):
		convert_lambda = (1/self.lam(xs))**(1/self.p)
		weibull = convert_lambda.view(-1)*np.random.weibull(self.p, size=xs.shape[0])
		weibull = weibull.reshape(-1,1)
		return weibull

	def mode(self, xs):
		convert_lambda = (1/self.lam(xs))**(1/self.p)
		return convert_lambda*((((self.p-1)/self.p))**(1/self.p))

	def mean(self, xs):
		convert_lambda = (1/self.lam(xs))**(1/self.p)
		return  convert_lambda*scipy.special.gamma(1. + 1./self.p)


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	d = 2
	p = 2
	lam = lambda x: torch.exp(torch.sum(x, dim = 1))
	lam_form = lambda x,theta: torch.exp(x@theta)

	W = WeibullNoise(lam, p = p, lam_form=lam_form)

	tstar = torch.ones(size = (2,1)).double()
	x = torch.ones(size = (1,2)).double()
	print(lam(x), lam_form(x,tstar))
	pdf = lambda y: W.noise_likelihood(y,x,tstar)#torch.exp(W.noise_log_likelihood(y,x,tstar))

	y = torch.linspace(0,5,1000).double()
	#plt.plot(y, pdf(y))
	samples = []
	mean = float(np.log(lam(x)))
	for _ in range(10000):
		samples.append(-np.log(float(W.sample_noise(x).view(-1)))*p - np.euler_gamma - mean)

	print (np.mean(samples))
	print( (np.pi**2/6))
	print (np.var(samples))
	#plt.plot(np.exp(W.mode(x)),pdf(W.mode(x)),'ko')

	plt.hist(samples, density=True)
	plt.show()


