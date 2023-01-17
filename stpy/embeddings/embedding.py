__author__ = "Mojmir Mutny"
__copyright__ = "Copyright (c) 2018 Mojmir Mutny, ETH Zurich"
__credits__ = ["Mojmir Mutny", "Andreas Krause"]
__license__ = "MIT Licence"
__version__ = "0.3"
__email__ = "mojmir.mutny@inf.ethz.ch"
__status__ = "DEV"

"""
This file implements code used in paper:

	Mojmir Mutny & Andreas Krause, "Efficient High Dimensional Bayesian Optimization 
	with Additivity and Quadrature Fourier Features", NIPS 2018

Namely, we implement finite basis approximation to Gaussian processes. The main contribution of this paper 
is implementation of the method embed(x) which coincides with \Phi(x) in product approximation 
	k(x,y) = \Phi(x)^\top \Phi(y)
"""

"""
Copyright (c) 2018 Mojmir Mutny, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List, Callable

import numpy as np
import torch
from scipy.stats import chi
from scipy.stats import norm

import stpy.helpers.helper as helper
import stpy.helpers.quadrature_helper as quad_help


class Embedding():
	"""
	Base class for Embeddings to approximate kernels with a higher dimensional linear product.
	"""

	def __init__(self, gamma=0.1, nu=0.5, m=100, d=1, diameter=1.0, groups=None, kappa=1.0,
				 kernel="squared_exponential", cosine=False, approx="rff", **kwargs):
		"""
		Called to calculate the embedding weights (either via sampling or deterministically)

		Args:
			gamma: (positional, 0.1) bandwidth of the squared exponential kernel
			nu: (positional, 0.5) the parameter of Matern family
			m: (positional, 1)
			d: dimension of the

		Returns:
			None
		"""
		self.gamma = float(gamma)
		self.n = nu
		self.m = int(m)
		self.d = int(d)
		self.nu = nu
		self.kappa = kappa
		self.cosine = cosine
		self.diameter = diameter
		self.groups = groups
		self.kernel = kernel
		self.approx = approx
		self.gradient_avail = 0
		if self.m % 2 == 1:
			raise AssertionError("Number of random features has to be even.")

	def sample(self):
		"""
		Called to calculate the embedding weights (either via sampling or deterministically)

		Args:
		    None

		Returns:
			None
		"""
		raise AttributeError("Only derived classes can call this method.")

	def embed(self, x):
		"""
		Called to calculate the embedding weights (either via sampling or deterministically)

		Args:
		    x: numpy array containing the points to be embedded in the format (n,d)

		Returns:
			y: numpy array containg the embedded points (n,m), where m is the embedding dimension
		"""

		raise AttributeError("Only derived classes can call this method.")

	def get_m(self):
		"""

		:return:

		"""
		return self.m

	def integral(self, S):
		a = S.bounds[:, 0]
		b = S.bounds[:, 1]
		psi = torch.zeros(self.m).double()

		for i in range(self.m // 2):
			omegas = self.W[i, :].view(-1)
			psi[i] = quad_help.integrate_cos_multidimensional(a.numpy(), b.numpy(), omegas.numpy())
			psi[self.m // 2 + i] = quad_help.integrate_sin_multidimensional(a.numpy(), b.numpy(), omegas.numpy())
		return psi


"""
===============================
	Sampling Based Methods
===============================
"""


class RFFEmbedding(Embedding):
	"""
		Random Fourier Features emebedding
	"""

	def __init__(self, biased=False, **kwargs):
		super().__init__(**kwargs)
		self.biased = biased
		self.sample()

	def sampler(self, size):
		"""
			Defines the sampler object

		Args:
		 	size:

		Return:
		"""
		if self.kernel == "squared_exponential":
			distribution = lambda size: np.random.normal(size=size) * (1. / self.gamma)
			inv_cum_dist = lambda x: norm.ppf(x) * (1. / self.gamma)

		elif self.kernel == "laplace":
			distribution = None
			inv_cum_dist = lambda x: (np.tan(np.pi * x - np.pi) / self.gamma)

		elif self.kernel == "modified_matern":
			if self.nu == 2:
				distribution = None
				inv_cum_dist = None
				pdf = lambda x: np.prod(2 * (self.gamma) / (np.power((1. + self.gamma ** 2 * x ** 2), 2) * np.pi),
										axis=1)
			elif self.nu == 3:
				distribution = None
				inv_cum_dist = None
				pdf = lambda x: np.prod((8. * self.gamma) / (np.power((1. + self.gamma ** 2 * x ** 2), 3) * 3 * np.pi),
										axis=1)
			elif self.nu == 4:
				distribution = None
				inv_cum_dist = None
				pdf = lambda x: np.prod((16. * self.gamma) / (np.power((1. + self.gamma ** 2 * x ** 2), 4) * 5 * np.pi),
										axis=1)

		# Random Fourier Features
		if self.approx == "rff":
			if distribution == None:
				if inv_cum_dist == None:
					self.W = helper.rejection_sampling(pdf, size=size)
				else:
					self.W = helper.sample_custom(inv_cum_dist, size=size)
			else:
				self.W = distribution(size)

		# Quasi Fourier Features
		elif self.approx == "halton":
			if inv_cum_dist != None:
				self.W = helper.sample_qmc_halton(inv_cum_dist, size=size)
			else:
				raise AssertionError("Inverse Cumulative Distribution could not be deduced")

		elif self.approx == "orf":
			distribution = lambda size: np.random.normal(size=size) * (1.)
			self.W = distribution(size)

			# QR decomposition
			self.Q, _ = np.linalg.qr(self.W)
			# df and size
			self.S = np.diag(chi.rvs(size[1], size=size[0]))
			self.W = np.dot(self.S, self.Q) / self.gamma ** 2

		return self.W

	def sample(self):
		"""
			Samples Random Fourier Features
		"""
		self.W = self.sampler(size=(self.m, self.d))
		self.W = torch.from_numpy(self.W)

		if self.biased == True:
			self.b = 2. * np.pi * np.random.uniform(size=(self.m))
			self.bs = self.b.reshape(self.m, 1)
			self.b = torch.from_numpy(self.b)
			self.bs = torch.from_numpy(self.bs)

	def embed(self, x):
		"""
		:param x: torch array
		:return: embeded vector
		"""
		(times, d) = x.shape
		if self.biased == True:
			z = np.sqrt(2. / self.m) * torch.t(torch.cos(self.W[:, 0:d].mm(torch.t(x)) + self.b.view(self.m, 1)))
		else:
			q = self.W[:, 0:d].mm(torch.t(x))
			# z[0:int(self.m / 2), :] = \
			z1 = np.sqrt(2. / float(self.m)) * torch.cos(q[0:int(self.m / 2), :])
			# z[int(self.m / 2):self.m, :] = np.sqrt(2. / float(self.m)) * torch.sin(q[int(self.m / 2):self.m, :])
			z2 = np.sqrt(2. / float(self.m)) * torch.sin(q[int(self.m / 2):self.m, :])
			z = torch.cat([z1, z2])

		return torch.t(z) * np.sqrt(self.kappa)


"""
===============================
	Quadrature Based Methods
===============================
"""


class QuadratureEmbedding(Embedding):
	"""
		General quadrature embedding
	"""

	def __init__(self, scale=1.0, **kwargs):
		Embedding.__init__(self, **kwargs)
		self.scale = scale
		self.compute()

	def reorder_complexity(self, omegas, weights):
		abs_omegas = np.abs(omegas)
		order = np.argsort(abs_omegas)
		new_omegas = omegas[order]
		new_weights = weights[order]
		return new_omegas, new_weights

	def derivative_1(self, x):
		(times, d) = tuple(x.size())
		# z = torch.from_numpy(np.zeros(shape=(self.m, times),dtype=x.dtype))
		z = torch.zeros(self.d, self.m, times, dtype=x.dtype)
		q = torch.mm(self.W[:, 0:d], torch.t(x))  # (m,d)x(d,n)

		omegas = self.W[:, 0:d]  # (m,d)

		if self.cosine == False:
			z[:, 0:int(self.m / 2), :] = -torch.einsum('ij,ik->jik', omegas,
													   torch.sqrt(self.weights.view(-1, 1)) * torch.sin(
														   q))  # (m,d)  (m,n)
			z[:, int(self.m / 2):self.m, :] = torch.einsum('ij,ik->jik', omegas,
														   torch.sqrt(self.weights.view(-1, 1)) * torch.cos(q))
		else:
			raise NotImplementedError("Cosine only features derivative not implemented")

		return np.sqrt(self.kappa) * z

	def derivative_2(self, x):
		(times, d) = tuple(x.size())
		# z = torch.from_numpy(np.zeros(shape=(self.m, times),dtype=x.dtype))
		z = torch.zeros(self.d, self.d, self.m, times, dtype=x.dtype)
		q = torch.mm(self.W[:, 0:d], torch.t(x))  # (m,d)x(d,n)

		omegas = self.W[:, 0:d]  # (m,d)

		if self.cosine == False:
			z[:, :, 0:int(self.m / 2), :] = -torch.einsum('il,ij,ik->jlik', omegas, omegas,
														  torch.sqrt(self.weights.view(-1, 1)) * torch.cos(
															  q))  # (m,d)  (m,d)  (m,n)
			z[:, :, int(self.m / 2):self.m, :] = -torch.einsum('il,ij,ik->jlik', omegas, omegas,
															   torch.sqrt(self.weights.view(-1, 1)) * torch.sin(q))
		else:
			raise NotImplementedError("Cosine only features derivative not implemented")

		return np.sqrt(self.kappa) * z

	def product_integral(self, S):
		"""
		Compute the outer product integral
		:param S: Borel set
		:return: m times m matrix with integrate entries
		"""
		assert S.d == 1 or S.d == 2
		if S.d == 1:
			a = S.bounds[0, 0]
			b = S.bounds[0, 1]
			h = self.m // 2
			Psi = torch.zeros(size=(self.m, self.m)).double()
			for i in range(h):
				for j in range(h):
					Psi[i, j] = torch.sqrt(self.weights[i] * self.weights[j]) * quad_help.integrate_cos_cos(a, b,
																											self.W[
																												i, 0],
																											self.W[
																												j, 0])  # cos cos
					Psi[i, j + h] = torch.sqrt(self.weights[i] * self.weights[j]) * quad_help.integrate_sin_cos(a, b,
																												self.W[
																													i, 0],
																												self.W[
																													j, 0])  # cos sin
					Psi[i + h, j] = torch.sqrt(self.weights[j] * self.weights[i]) * quad_help.integrate_sin_cos(a, b,
																												self.W[
																													j, 0],
																												self.W[
																													i, 0])  # sin cos
					Psi[i + h, j + h] = torch.sqrt(self.weights[i] * self.weights[j]) * quad_help.integrate_sin_sin(a,
																													b,
																													self.W[
																														i, 0],
																													self.W[
																														j, 0])  # sin sin
		elif S.d == 2:
			xa = S.bounds[0, 0]
			xb = S.bounds[0, 1]
			ya = S.bounds[1, 0]
			yb = S.bounds[1, 1]
			h = self.m // 2
			Psi = torch.zeros(size=(self.m, self.m)).double()
			for i in range(h):
				for j in range(h):
					Psi[i, j] = torch.sqrt(self.weights[i] * self.weights[j]) \
								* quad_help.integrate2d_cos_cos(xa, ya, xb, yb, self.W[i, 0], self.W[i, 1],
																self.W[j, 0], self.W[j, 1])  # cos cos
					Psi[i, j + h] = torch.sqrt(self.weights[i] * self.weights[j]) \
									* quad_help.integrate2d_sin_cos(xa, ya, xb, yb, self.W[i, 0], self.W[i, 1],
																	self.W[j, 0], self.W[j, 1])  # cos cos
					Psi[i + h, j] = torch.sqrt(self.weights[j] * self.weights[i]) \
									* quad_help.integrate2d_sin_cos(xa, ya, xb, yb, self.W[j, 0], self.W[j, 1],
																	self.W[i, 0], self.W[i, 1])  # cos cos
					Psi[i + h, j + h] = torch.sqrt(self.weights[i] * self.weights[j]) \
										* quad_help.integrate2d_sin_sin(xa, ya, xb, yb, self.W[i, 0], self.W[i, 1],
																		self.W[j, 0], self.W[j, 1])  # cos cos
		return self.kappa * Psi

	def compute(self, complexity_reorder=True):
		"""
			Computes the tensor grid for Fourier features
		:return:
		"""

		if self.cosine == False:
			self.q = int(np.power(self.m // 2, 1. / self.d))
			self.m = self.q ** self.d
		else:
			self.q = int(np.power(self.m, 1. / self.d))
			self.m = self.q ** self.d

		(omegas, weights) = self.nodesAndWeights(self.q)

		if complexity_reorder == True:
			(omegas, weights) = self.reorder_complexity(omegas, weights)

		self.weights = helper.cartesian([weights for weight in range(self.d)])
		self.weights = np.prod(self.weights, axis=1)

		v = [omegas for omega in range(self.d)]
		self.W = helper.cartesian(v)

		if self.cosine == False:
			self.m = self.m * 2
		else:
			pass

		self.W = torch.from_numpy(self.W)
		self.weights = torch.from_numpy(self.weights)

	def transform(self):
		"""

		:return: spectral density of a kernel
		"""
		if self.kernel == "squared_exponential":
			p = lambda omega: np.exp(-np.sum(omega ** 2, axis=1).reshape(-1, 1) / 2 * (self.gamma ** 2)) * np.power(
				(self.gamma / np.sqrt(2 * np.pi)), 1.) * np.power(np.pi / 2, 1.)

		elif self.kernel == "laplace":
			p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.), axis=1).reshape(-1, 1) * np.power(
				self.gamma / 2., 1.)

		elif self.kernel == "modified_matern":
			if self.nu == 2:
				p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,
																												   1) * np.power(
					self.gamma * 1, 1.)
			elif self.nu == 3:
				p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,
																												   1) * np.power(
					self.gamma * 4 / 3, 1.)
			elif self.nu == 4:
				p = lambda omega: np.prod(1. / ((self.gamma ** 2) * (omega ** 2) + 1.) ** self.nu, axis=1).reshape(-1,
																												   1) * np.power(
					self.gamma * 8 / 5, 1.)

		return p

	def nodesAndWeights(self, q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""

		# For osciallatory integrands even this has good properties.
		# weights = np.ones(self.q) * self.scale * np.pi / (self.q + 1)
		# omegas = (np.linspace(0, self.q - 1, self.q)) + 1
		# omegas = omegas * (np.pi / (self.q + 1))

		(omegas, weights) = np.polynomial.legendre.leggauss(2 * q)

		omegas = omegas[q:]
		weights = 2 * weights[q:]

		omegas = ((omegas + 1.) / 2.) * np.pi
		sine_scale = (1. / (np.sin(omegas) ** 2))
		omegas = self.scale / np.tan(omegas)
		prob = self.transform()
		weights = self.scale * sine_scale * weights * prob(omegas.reshape(-1, 1)).flatten()
		return (omegas, weights)

	def embed(self, x):
		"""
		:param x: torch array
		:return: embeding of the x
		"""
		(times, d) = tuple(x.size())
		# z = torch.from_numpy(np.zeros(shape=(self.m, times),dtype=x.dtype))
		z = torch.zeros(self.m, times, dtype=x.dtype)
		q = torch.mm(self.W[:, 0:d], torch.t(x))

		if self.cosine == False:
			z[0:int(self.m / 2), :] = torch.sqrt(self.weights.view(-1, 1)) * torch.cos(q)
			z[int(self.m / 2):self.m, :] = torch.sqrt(self.weights.view(-1, 1)) * torch.sin(q)
		else:
			z = torch.sqrt(self.weights.view(-1, 1)) * torch.cos(q)

		return torch.t(z) * np.sqrt(self.kappa)

	def get_sub_indices(self, group):
		"""
		:param group: group part of the embeding to embed
		:return: embeding of x in group
		"""
		m2 = self.m
		mhalf = int(np.power(self.m // 2, 1. / self.d))

		m = 2 * mhalf
		mquater = mhalf // 2

		if group == 0:
			ind = np.arange(mquater * mhalf, (mquater + 1) * mhalf, 1).tolist() + np.arange(m2 // 2 + (mquater * mhalf),
																							m2 // 2 + (
																										mquater + 1) * mhalf,
																							1).tolist()
			return ind
		else:
			ind = np.arange(mquater, m2 // 2, mhalf).tolist() + np.arange(m2 // 2 + mquater, m2, mhalf).tolist()
			return ind

	def get_sum_sub_indices(self, group):

		# idenitfy unique values
		arr = self.W[:, group]
		values = np.unique(arr)
		# find indices of each unique value
		ind = []
		for value in values:
			ind_inside = []
			for index, elem in enumerate(arr):
				if elem == value:
					ind_inside.append(index)
			ind.append(ind_inside)
			ind_inside2 = [i + self.m // 2 for i in ind_inside]
			ind.append(ind_inside2)
		return ind


class TrapezoidalEmbedding(QuadratureEmbedding):

	def __init__(self, **kwargs):
		QuadratureEmbedding.__init__(self, **kwargs)
		if self.kernel != "squared_exponential":
			raise AssertionError("This embeding is allowed only with Squared Exponential Kernel")

	def nodesAndWeights(self, q):
		prob = self.transform()
		# prob = lambda x:
		h = np.sqrt(np.pi / q) / self.gamma ** 2

		nodes = np.linspace(-q // 2, q // 2, q) * h
		# print (nodes)

		weights = h * prob(nodes.reshape(-1, 1)).flatten() * (2 / np.pi)

		# nodes = np.sqrt(2) * nodes / self.gamma

		return (nodes, weights)


class ClenshawCurtisEmbedding(QuadratureEmbedding):

	def __init__(self, **kwargs):
		QuadratureEmbedding.__init__(self, **kwargs)
		if self.kernel != "squared_exponential":
			raise AssertionError("This embeding is allowed only with Squared Exponential Kernel")

	def nodesAndWeights(self, q):
		L = 1. / self.gamma
		prob = self.transform()
		# prob = lambda x:

		nodes_0 = np.linspace(0, q + 1, q + 2)
		nodes_0 = np.pi * nodes_0[1:-1] / (q + 2)
		nodes = L / np.tan(nodes_0)

		weights = L * (np.pi / (q + 2)) * (1. / np.sin(nodes_0) ** 2)
		weights = weights * prob(nodes.reshape(-1, 1)).flatten() * (2. / np.pi)

		return (nodes, weights)

	def nodesAndWeights2(self, q):
		prob = self.transform()

		nodes_0 = np.linspace(0, q + 1, q + 2)
		nodes_0 = nodes_0[1:-1] / (q + 2) * np.pi

		nodes = np.sqrt(-np.log(np.sin(nodes_0[0:q // 2])))
		nodes2 = -np.sqrt(-np.log(np.sin(nodes_0[q // 2:])))

		n1 = nodes_0[0:q // 2]
		n2 = nodes_0[q // 2:]

		weights = (1. / np.tan(n1)) * (1. / np.sqrt(-np.log(np.sin(n1)))) * prob(
			nodes.reshape(-1, 1)).flatten() * np.pi / (q + 2)
		weights2 = -(1. / np.tan(n2)) * (1. / np.sqrt(-np.log(np.sin(n2)))) * prob(
			nodes.reshape(-1, 1)).flatten() * np.pi / (q + 2)

		nodes = np.concatenate((nodes, nodes2))
		weights = np.concatenate((weights, weights2))

		return (nodes, weights)


class HermiteEmbedding(QuadratureEmbedding):
	"""
		Hermite Quadrature Fourier Features for squared exponential kernel
	"""

	def __init__(self, ones=False, cosine=False, **kwargs):
		self.ones = ones
		self.cosine = cosine
		QuadratureEmbedding.__init__(self, **kwargs)
		if self.kernel != "squared_exponential":
			raise AssertionError("Hermite Embedding is allowed only with Squared Exponential Kernel")

	def nodesAndWeights(self, q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""
		(nodes, weights) = np.polynomial.hermite.hermgauss(2 * q)
		# print (nodes)
		nodes = nodes[q:]
		weights = 2 * weights[q:]

		if self.ones == True:
			weights = np.ones(q)

		nodes = np.sqrt(2) * nodes / self.gamma
		weights = weights / np.sqrt(np.pi)
		return (nodes, weights)


class OverCompleteHermiteEmbedding(HermiteEmbedding):

	def nodesAndWeights(self, q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""
		(nodes, weights) = np.polynomial.hermite.hermgauss(q)
		nodes = nodes
		weights = weights

		nodes = np.sqrt(2) * nodes / self.gamma
		weights = weights / np.sqrt(np.pi)
		return (nodes, weights)


class MaternEmbedding(QuadratureEmbedding):
	"""
		Matern specific quadrature based Fourier Features
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		if self.kernel != "modified_matern" and self.kernel != "laplace":
			raise AssertionError("Matern Embedding is allowed only with Matern Kernel")

	def nodesAndWeights(self, q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""
		(nodes, weights) = np.polynomial.hermite.hermgauss(q)
		nodes = np.sqrt(2) * nodes / self.gamma
		weights = weights / np.sqrt(np.pi)
		return (nodes, weights)


class QuadPeriodicEmbedding(QuadratureEmbedding):
	"""
		General class implementing
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def nodesAndWeights(self, q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""
		weights = np.ones(self.q) * self.scale * 2 / (self.q + 1)
		omegas = (np.linspace(0, self.q - 1, self.q)) + 1
		omegas = omegas * (np.pi / (self.q + 1))

		sine_scale = (1. / (np.sin(omegas) ** 2))
		omegas = self.scale / np.tan(omegas)
		prob = self.transform()
		weights = self.scale * sine_scale * weights * prob(omegas.reshape(-1, 1)).flatten()
		return (omegas, weights)


class KLEmbedding(QuadratureEmbedding):
	"""
		General class implementing Karhunen-Loeve expansion
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)


class LatticeEmbedding(QuadratureEmbedding):
	"""
		Class for standard basis indexed by natural numbers
	"""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	# if self.kernel != "modified_matern" and self.kernel !="laplace":
	#	raise AssertionError("Matern Embedding is allowed only with Matern Kernel")

	def nodesAndWeights(self, q):
		"""
		Compute nodes and weights of the quadrature scheme in 1D

		:param q: degree of quadrature
		:return: tuple of (nodes, weights)
		"""
		nodes = np.arange(1, q + 1, 1)
		nodes = np.sqrt(2) * nodes / self.gamma
		weights = np.ones(q) / (2 * q)
		return (nodes, weights)


class ConcatEmbedding(Embedding):

	def __init__(self, embeddings: List[Embedding]):

		self.embeddings = embeddings
		self.m = sum([emb.get_m() for emb in embeddings])

	def embed(self, xtest):
		return torch.hstack([emb.embed(xtest) for emb in self.embeddings])


class MaskedEmbedding(Embedding):

	def __init__(self, embedding: Embedding, mask: Callable):
		self.embedding = embedding
		self.m = self.embedding.get_m()
		self.mask = mask

	def embed(self, xtest):
		return torch.diag(self.mask(xtest))@self.embedding.embed(xtest)


class AdditiveEmbeddings():

	def __init__(self, embeddings, ms, groups=None, scaling=None, additive=True):
		self.emebeddings = embeddings
		if scaling is None:
			self.scaling = torch.ones(len(self.emebeddings)).double()  # /np.sqrt(len(self.emebeddings))
		else:
			self.scaling = scaling
		self.additive = additive

		if groups is not None:
			self.groups = groups
		else:
			self.groups = [[i] for i in range(len(self.emebeddings))]

		try:
			self.ms = torch.Tensor(ms)
		except:
			self.ms = ms

		self.no_emb = len(self.emebeddings)
		self.m  = torch.sum(self.ms)

	def embed(self, x):
		if self.additive:
			r = torch.zeros(size=(x.size()[0], int(torch.sum(self.ms)))).double()
			count = 0
			for index, embedding in enumerate(self.emebeddings):
				r[:, count:count + int(self.ms[index])] = \
					embedding.embed(x[:, self.groups[index]].view(-1, len(self.groups[index]))) * self.scaling[index]
				count = count + int(self.ms[index])
			return r
		else:
			pass


class ProjectiveEmbeddings():

	def __init__(self, embedding, project):
		self.embedding = embedding
		self.project = project

	def embed(self, x):
		r = self.embedding.embed(self.project(x))
		return r
