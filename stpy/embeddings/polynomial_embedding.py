__author__ = "Mojmir Mutny"
__copyright__ = "Copyright (c) 2018 Mojmir Mutny, ETH Zurich"
__credits__ = ["Mojmir Mutny", "Andreas Krause"]
__license__ = "MIT Licence"
__version__ = "0.2"
__email__ = "mojmir.mutny@inf.ethz.ch"
__status__ = "DEV"

"""
This file implements a polynomial embedding 
	k(x,y) = \Phi(x)^\top \Phi(y)
	for kernels of the form (x^\top y + 1)^p
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

import numpy as np
import numpy.polynomial.chebyshev as cheb
import scipy.integrate as integrate
import torch
from sklearn.preprocessing import PolynomialFeatures


class CustomEmbedding():
	def __init__(self, d, embedding_function, m, groups=None, quadrature="fixed"):
		self.d = d
		self.groups = groups
		self.embedding_function = embedding_function
		self.m = m
		self.quadrature = quadrature

	def embed(self, x):
		return self.embedding_function(x)

	def get_m(self):
		return self.m

	def integral(self, S):
		varphi = torch.zeros(size=(self.m, 1)).double()

		if self.quadrature == "fixed":
			if S.d == 1:
				weights, nodes = S.return_legendre_discretization(n=512)
				Z = self.embed(nodes)
				varphi = torch.einsum('i,ij->j', weights, Z)
				return varphi.view(-1, 1)
			elif S.d == 2:
				weights, nodes = S.return_legendre_discretization(n=50)
				Z = self.embed(nodes)
				varphi = torch.einsum('i,ij->j', weights, Z)
				return varphi.view(-1, 1)
		else:
			if S.d == 1:
				for i in range(self.m):
					Fi = lambda x: self.embed(torch.from_numpy(np.array(x)).view(1, -1)).view(-1).numpy()
					val, status = integrate.quad(Fi, float(S.bounds[0, 0]), float(S.bounds[0, 1]))
					varphi[i] = val
			elif S.d == 2:
				for i in range(self.m):
					Fi = lambda x: self.embed(x).view(-1)[i]
					integrand = lambda x, y: Fi(torch.Tensor([x, y]).view(1, 2).double()).numpy()
					val, status = integrate.dblquad(integrand, float(S.bounds[0, 0]), float(S.bounds[0, 1]),
													lambda x: float(S.bounds[1, 0]),
													lambda x: float(S.bounds[1, 1]), epsabs=1.49e-03, epsrel=1.49e-03)
					varphi[i] = val
			return varphi


class PolynomialEmbedding():

	def __init__(self, d, p, kappa=1., groups=None, include_bias=True):
		self.d = d
		self.p = p
		self.kappa = kappa
		self.groups = groups
		self.compute(include_bias=include_bias)
		self.include_bias = include_bias

	def compute(self, include_bias=True):
		self.poly = PolynomialFeatures(degree=self.p, include_bias=include_bias)
		if self.groups is None:
			self.poly.fit_transform(np.random.randn(1, self.d))
			self.degrees = torch.from_numpy(self.poly.powers_).double()
			self.size = self.degrees.size()[0]
		else:
			self.degrees = []
			self.size = 0
			self.sizes = []
			for group in self.groups:
				self.poly.fit_transform(np.random.randn(1, len(group)))
				z = torch.from_numpy(self.poly.powers_).double()
				self.degrees.append(z)
				self.sizes.append(z.size()[0])
				self.size += z.size()[0]

	def embed_group(self, x, j):
		(n, d) = x.size()
		x = x.view(n, -1)
		Phi = torch.zeros(size=(n, self.sizes[j]), dtype=torch.float64)
		group = self.groups[j]
		for i in range(n):
			y = x[i, :]
			z = y.view(1, len(group))
			Phi[i, :] = torch.prod(torch.pow(z, self.degrees[j]), dim=1).view(-1)
		return Phi

	def get_sub_indices(self, group):
		ind = []
		for index, elem in enumerate(self.degrees):
			z = torch.sum(elem[0:group - 2]) + torch.sum(elem[group + 1:])
			if (elem[group] >= 0.0) and (z <= 0.):
				ind.append(index)
		return ind

	def embed(self, x):
		(n, d) = x.size()
		# zero = torch.pow(x[0,:] * 0, self.degrees)
		Phi = torch.zeros(size=(n, self.size), dtype=torch.float64)

		if self.groups is None:
			for i in range(n):
				y = x[i, :]
				Phi[i, :] = torch.prod(torch.pow(y, self.degrees), dim=1)
		else:
			for i in range(n):
				y = x[i, :]
				for j, group in enumerate(self.groups):
					z = y[group].view(1, len(group))
					start = int(np.sum(self.sizes[0:j]))
					end = np.sum(self.sizes[0:j + 1])
					Phi[i, start:end] = torch.prod(torch.pow(z, self.degrees[j]), dim=1).view(-1)
		return np.sqrt(self.kappa) * Phi

	def derivative_1(self, x):
		pass

	def derivative_2(self, x):
		pass


class ChebyschevEmbedding():


	def get_m(self):
		return self.m

	def __init__(self, d, p, groups=None, include_bias=True):
		self.d = d
		self.p = p
		self.groups = groups
		self.c = np.ones(self.p)
		self.poly = cheb.Chebyshev(self.c)
		self.size = self.p
		self.m = self.p

	def embed(self, x):
		out = np.zeros(shape=(int(x.size()[0]), self.p))
		z = None
		for p in np.arange(1, self.p + 1, 1):
			c = np.ones(p)
			if p > 1:
				zold = z
				z = cheb.chebval(x.numpy(), c)
				out[:, p - 1] = (z - zold).reshape(-1)
			else:
				z = cheb.chebval(x.numpy(), c)
				out[:, p - 1] = z.reshape(-1)
		return torch.from_numpy(out)

	def derivative_1(self, x):
		pass

	def derivative_2(self, x):
		pass


if __name__ == "__main__":
	d = 2
	p = 4
	emb = PolynomialEmbedding(d, p, groups=[[0], [1]])
	x1 = torch.randn(size=(1, d), dtype=torch.float64)
	x2 = torch.randn(size=(1, d), dtype=torch.float64)
	xc = torch.cat((x1, x2))

	print(emb.embed(x1).size())
	print(emb.embed(x2).size())
	print(emb.embed(xc).size())

	print("--------")
	emb = PolynomialEmbedding(d, p)
	print(emb.get_sub_indices(0))
# d = 1
# emb = ChebyschevEmbedding(d,3)
# x1 = torch.randn(size = (1,d), dtype = torch.float64)
# x2 = torch.randn(size = (1,d), dtype = torch.float64)
# xc = torch.cat((x1,x2))
#
# print (xc)
# print (emb.embed(x1).size())
# print (emb.embed(x2).size())
# print (emb.embed(xc).size())
#
# print (emb.embed(xc))
