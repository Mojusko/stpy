from stpy.embeddings.embedding import Embedding
import stpy.helpers.helper as helper
from scipy.special import comb
from sklearn.preprocessing import PolynomialFeatures
import torch
import numpy as np

class Transformation(Embedding):

	def __init__(self):
		pass

	def embed(self,x):
		pass

	def linear_embedding(self):
		embed = lambda x: x
		return embed

	def create_polynomial_embeding(self, degree, d, kappa = 1., bias = False):
		"""
		create polynomial embeding

		:param degree:
		:param d:
		:return:
		"""
		m = int(comb(degree + d-1,degree-1)) + int(bias)
		poly = PolynomialFeatures(degree, include_bias = bias)
		embed = lambda x: kappa*torch.from_numpy(poly.fit_transform(x.numpy()))
		return embed, m
		return (nodes, weights)

	def embed(self,x):
		(times, d) = tuple(x.size())
		# z = torch.from_numpy(np.zeros(shape=(self.m, times),dtype=x.dtype))
		z = torch.zeros(self.m, times, dtype=x.dtype)
		q = torch.mm(self.W[:, 0:d], torch.t(x))
		z[0:int(self.m / 2), :] = torch.cos(q)
		z[int(self.m / 2):self.m, :] =  torch.sin(q)
		return torch.t(z)

	def create_fourier_embeding(self, cutoff, d, domain, bias = False):
		self.m = 2*cutoff - 2*int(bias)
		self.d = d
		omegas = np.arange(int(bias),cutoff,1)*2.*np.pi/(2*domain)
		print (omegas)
		v = [omegas for omega in range(self.d)]
		self.W = torch.from_numpy(helper.cartesian(v))
		embed = lambda x: self.embed(x)
		return embed, self.m


	def create_cosine_embeding(self, cutoff, d, domain):
		self.m = cutoff
		self.d = d
		omegas = np.arange(0,cutoff,1)*2.*np.pi/(2*domain)
		print (omegas)
		v = [omegas for omega in range(self.d)]
		self.W = torch.from_numpy(helper.cartesian(v))
		embed = lambda x:  torch.t(torch.cos(torch.mm(self.W[:, 0:d], torch.t(x))))
		return embed, self.m

	def create_cosine_power_embeding(self, cutoff, d, domain):
		self.m = cutoff+1
		self.d = d
		print (np.logspace(0,cutoff,num = cutoff+1, base = 2))
		omegas = np.logspace(0,cutoff,num = cutoff+1, base = 2)*2.*np.pi/(2*domain)
		print (omegas)
		v = [omegas for omega in range(self.d)]
		self.W = torch.from_numpy(helper.cartesian(v))
		embed = lambda x:  torch.t(torch.cos(torch.mm(self.W[:, 0:d], torch.t(x))))
		return embed, self.m
