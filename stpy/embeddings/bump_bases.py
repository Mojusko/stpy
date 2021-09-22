import copy
import torch
import numpy as np
import scipy
from scipy import integrate
import cvxpy as cp
from stpy.helpers.helper import cartesian
from stpy.embeddings.positive_embedding import PositiveEmbedding
from stpy.kernels import KernelFunction
from scipy.interpolate import BPoly
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly
from stpy.borel_set import BorelSet
from stpy.continuous_processes.nystrom_fea import NystromFeatures

class TriangleEmbedding(PositiveEmbedding):

	def __init__(self,*args,**kwargs):

		super().__init__(*args,**kwargs)


	def basis_fun(self, x, j):
		"""
		Return the value of basis function \phi_j(x)

		:param x: double, need to be in the interval
		:param j: integer, index of hat functions, 0 <= j <= m-1
		:return: \phi_j(x)
		"""

		dm = (self.interval[1] - self.interval[0]) / (self.m - 1)  # delta m
		tj = self.interval[0] + (j) * dm
		res = 1 - torch.abs((x - tj) / dm)
		res[res < 0] = 0
		return res

	def integrate_1d(self, a,b,tj,dm):
		"""
		:param a: from
		:param b: to
		:param tj: node
		:param dm: width
		:return:
		"""
		if a <= tj - dm and b >= tj + dm:  # contained
			vol = 1. * dm

		elif a >= tj + dm or b <= tj - dm:  # outside
			vol = 0.

		elif a <= tj - dm and b >= tj and b<=tj+dm:  # a out , b inside second half
			res = max(1. - np.abs((b - tj) / dm),0)
			vol = dm * 0.5 + (b-tj) * (1. + res) / 2.

		elif b >= tj + dm and a <= tj and a >= tj -dm: # b out, a inside first half
			res = max(1. - np.abs((a - tj) / dm),0)
			vol = dm * 0.5 + (tj-a) * (1. + res) / 2.

		elif a <= tj - dm and b <= tj and b>=tj -dm:  # a out, b inside first half
			res = max(1. - np.abs((b - tj) / dm),0)
			vol = 0.5 * (b-(tj-dm)) * res

		elif b >= tj + dm and a >= tj and a<=tj+dm: #b out, a inside second half
			res = max(1. - np.abs((a - tj) / dm),0)
			vol = 0.5 * ((tj + dm) - a) * res


		else:  # inside
			resa = max(1. - np.abs((a - tj) / dm),0)
			resb = max(1. - np.abs((b - tj) / dm),0)

			if b<=tj:
				vol = (b-a)*(resb+resa)/2.
			elif a>=tj:
				vol = (b-a)*(resa+resb)/2.
			else:
				vol = (tj - a) * (1+resa) / 2. + (b - tj) * (resb+1) / 2.


		return vol


	def integral(self, S):
		"""
		Integrate the Phi(x) over S
		:param S: borel set
		:return:
		"""
		if S in self.procomp_integrals.keys():
			return self.procomp_integrals[S]


		else:
			assert( S.d == self.d)
			psi = torch.zeros(self.get_m()).double()

			if self.d == 1:
				dm = (self.interval[1] - self.interval[0]) / (self.m -1)  # delta m
				a,b = S.bounds[0,0],S.bounds[0,1]
				for j in range(self.get_m()):
					tj = self.interval[0] + j * dm
					vol = self.integrate_1d(a.numpy(),b.numpy(),tj,dm)
					psi[j] = vol

			elif self.d == 2:
				dm = (self.interval[1] - self.interval[0]) / (self.m -1)  # delta m

				xa,xb = S.bounds[0,0],S.bounds[0,1]
				ya,yb = S.bounds[1,0],S.bounds[1,1]

				for j in range(self.get_m()):
					tj = self.interval[0] + (j%self.m) * dm
					tk = self.interval[0] + (j//self.m) * dm

					# triangle center point
					#center_point = torch.Tensor( [tj,tk]).view(-1,1)
					vol = self.integrate_1d(xa.numpy(),xb.numpy(),tk,dm)
					vol2 = self.integrate_1d(ya.numpy(),yb.numpy(),tj,dm)
					psi[j] = vol*vol2
					#if torch.sum(S.is_inside(center_point)):
					#psi[j] = (dm**2)/3.
			else:
				raise ("more than 2D not implemented.")

			Gamma_half = self.cov()
			emb = psi @ Gamma_half
			self.procomp_integrals[S] = emb
			return emb

	# def product_integral(self, S):
	# 	assert( S.d == self.d)
	# 	dm = (self.interval[1] - self.interval[0]) / (self.m - 1)  # delta m
	#
	# 	# for i in range(self.get_m()):
	# 	# 	for j in range(self.get_m()):
	# 	# 		if np.abs(j-i)>1:
	# 	# 			Psi[i,j] = 0
	# 	# 		elif np.abs(j-i) == 1:
	# 	# 			Psi[i,j] = 0
	# 	Psi =
	#
	# 	Psi = torch.eye(self.m).double()*dm/2 + ( torch.diag(torch.ones(self.get_m()-1),diagonal=1) +torch.diag(torch.ones(self.get_m()-1),diagonal=-1)).double() *dm/4
	# 	Gamma_half = self.cov()
	# 	return  Gamma_half.T @ Psi @ Gamma_half

class FaberSchauderEmbedding(TriangleEmbedding):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		if int(np.log2(self.m))!=np.log2(self.m):
			raise AssertionError("This basis works only with multiple of 2.")

	def basis_fun(self, x, j):
		"""
		Return the value of basis function \phi_j(x)

		:param x: double, need to be in the interval
		:param j: integer, index of hat functions, 0 <= j <= m-1
		:return: \phi_j(x)
		"""
		if j == 0:
			res = x*0 + 1
		elif j == 1:
			dm = (self.interval[1] - self.interval[0])/2  # delta m
			res = 1 - torch.abs((x) / dm)
			res[res < 0] = 0
		else:
			level = np.floor(np.log2(j))
			start = 2**level
			dm = (self.interval[1] - self.interval[0])/(2*start)
			tj = self.interval[0] + (j-start)*2*dm + dm
			res = 1 - torch.abs((x - tj) / dm)
			res[res < 0] = 0
		return res

	def integral(self,S):
		assert( S.d == self.d)
		psi = torch.zeros(self.get_m()).double()

		if self.d == 1:
			a,b = S.bounds[0,0],S.bounds[0,1]
			for j in range(self.get_m()):
				if j == 0:
					vol = (b-a)
				elif j == 1:
					dm = (self.interval[1] - self.interval[0]) / 2  # delta m
					vol = self.integrate_1d(a.numpy(), b.numpy(), 0, dm)
				else:
					level = np.floor(np.log2(j))
					start = 2 ** level
					dm = (self.interval[1] - self.interval[0]) / (2 * start)
					tj = self.interval[0] + (j - start) * 2 * dm + dm
					vol = self.integrate_1d(a.numpy(), b.numpy(), tj, dm)
				psi[j] = vol
		return psi

	def product_integral(self):
		raise NotImplementedError("Not implemented.")
		pass

class KuhnExponentialEmbedding(PositiveEmbedding):
	"""
	Basis from: Covering numbers of Gaussian reproducing kernel Hilbert spaces
	by Thomas Kuhn

	"""
	def __init__(self, *args,gamma = 0.1, **kwargs):
		super().__init__(self,*args, **kwargs)
		self.gamma = gamma

	def basis_fun(self, x, j):
		k = np.exp(j/2 * np.log(1./self.gamma) - (j/2)*scipy.special.gammaln(j+1))
		res = k*(x**j)*torch.exp(- (x**2)/(2*self.gamma**2))
		mask1 = x < 0
		mask2 = x > 1
		res[mask1] = 0.
		res[mask2] = 0.
		return res

class CustomHaarBumps(PositiveEmbedding):
	"""

	Custom Haar basis that cover different sized pockets of domain

	"""

	# def __init__(self, *args, **kwargs):
	# 	super().__init__(self,*args, **kwargs)
	# 	nodes = None
	# 	widths = None
	# 	self.nodes = nodes
	# 	self.widths = widths

	def __init__(self,d,m, nodes, widths,weights,**kwargs):
		super().__init__(d,m,**kwargs)
		self.nodes = nodes
		self.widths = widths
		self.weights = weights

	def basis_fun(self,x,j):

		if self.nodes is None or self.widths is None:
			super().basis_fun(x,j)
		else:
			mask = np.abs(x-self.nodes[j])<self.widths[j]
			out = x*0
			out[mask] = self.weights[j]
			return out

class BumpsEmbedding(PositiveEmbedding):

	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def integrate(self, a,b,j):
		vol = 0.
		return vol

	def integral(self, S):
		"""
		Integrate the Phi(x) over S
		:param S: borel set
		:return:
		"""
		assert( S.d == self.d)
		psi = torch.zeros(self.get_m()).double()

		a,b = S.bounds[0,0],S.bounds[0,1]
		for j in range(self.get_m()):
			vol = self.integrate(a.numpy(),b.numpy(),j)
			psi[j] = vol

	def basis_fun(self, x, j): #1d
		"""
		Return the value of basis function \phi_j(x)

		:param x: double, need to be in the interval
		:param j: integer, index of hat functions, 0 <= j <= m-1
		:return: \phi_j(x)
		"""

		dm = (self.interval[1] - self.interval[0]) / (self.m -1)  # delta m
		tj = self.interval[0] + (j) * dm
		res = -(x-tj)*(x-(tj+(2*dm)))*(1./(dm**2))
		res[res < 0] = 0
		return res

class PositiveNystromEmbedding(PositiveEmbedding):

	def __init__(self, *args,samples = 300, **kwargs):
		super().__init__(*args, **kwargs)
		self.samples = np.maximum(samples, self.m)

		B = BorelSet(1, torch.Tensor([[self.interval[0], self.interval[1]]]).double())
		x = B.return_discretization(256)
		y = x[:,0].view(-1,1)*0

		print ("Starting optimal basis construction, with m =",self.m)
		self.new_kernel_object = KernelFunction(kernel_name=self.kernel_object.optkernel,
												gamma = self.kernel_object.gamma)
		self.GP = NystromFeatures(self.new_kernel_object,m = self.m, approx = 'positive_svd',
								  samples = self.samples)
		self.GP.fit_gp(x,y)
		print ("Optimal basis constructed.")
		if torch.sum(torch.isnan(self.GP.embed(x)))>0:
			print ("Failed basis? (zero is good):",torch.sum(torch.isnan(self.GP.embed(x))))

		self.precomp_integral = {}

	def basis_fun(self,x,j):
		return self.GP.embed(x)[:,j].view(-1,1)

	def get_constraints(self):
		s = self.m**self.d
		l = np.full(s, 0.0).astype(float)
		u = np.full(s, 10e10)
		Lambda = np.identity(s)
		return (l,Lambda,u)

	def integral(self, S):
		assert( S.d == self.d)

		if S in self.precomp_integral.keys():
			return self.precomp_integral[S]
		else:
			if S.d == 1:
				weights, nodes = S.return_legendre_discretization(n=256)
				psi = torch.sum(torch.diag(weights) @ self.GP.embed(nodes), dim=0)
				Gamma_half = self.cov()
				psi = Gamma_half.T @ psi
				self.precomp_integral[S] = psi
			elif S.d == 2:
				weights, nodes = S.return_legendre_discretization(n=50)
				vals = self.embed_internal(nodes)
				psi = torch.sum(torch.diag(weights)@vals, dim = 0)
				Gamma_half = self.cov()
				psi = Gamma_half.T @ psi
				self.precomp_integral[S] = psi
				if  torch.sum(torch.isnan(psi))>0:
					print("Failed integrals? (0 is good):", torch.sum(torch.isnan(psi)))

			else:
				raise NotImplementedError("Higher dimension not implemented.")
			return  psi

	def cov(self, inverse=False):

		if self.precomp == False:

			x = torch.linspace(self.interval[0],self.interval[1],256)
			vals = self.GP.embed(x)
			indices = torch.argmax(vals, dim = 0)
			t = x[indices]

			if self.d == 1:
				t = t.view(-1, 1).double()
			elif self.d == 2:
				t = torch.from_numpy(cartesian([t.numpy(), t.numpy()])).double()
			elif self.d == 3:
				t = torch.from_numpy(cartesian([t.numpy(), t.numpy(), t.numpy()])).double()

			self.Gamma = self.kernel(t, t)
			Z = self.embed_internal(t)

			M = torch.pinverse(Z.T @ Z + (self.s) * torch.eye(self.Gamma.size()[0]))
			self.M = torch.from_numpy(np.real(scipy.linalg.sqrtm(M.numpy())))

			# self.Gamma_half = torch.cholesky(Gamma \
			#	+ self.s * self.s * torch.eye(Gamma.size()[0]).double(), upper = True	)

			self.Gamma_half = torch.from_numpy(
				np.real(scipy.linalg.sqrtm(self.Gamma.numpy() + (self.s ** 2) * np.eye(self.Gamma.size()[0]))))
			self.Gamma_half = self.M @ self.Gamma_half
			self.invGamma_half = torch.pinverse(self.Gamma_half)
			self.precomp = True
		else:
			pass

		if inverse == True:
			return self.Gamma_half, self.invGamma_half
		else:
			return self.Gamma_half






if __name__ == "__main__":
	from stpy.continuous_processes.gauss_procc import GaussianProcess
	from stpy.helpers.helper import interval
	import matplotlib.pyplot as plt

	d = 1
	m = 32
	n = 64
	N = 20
	sqrtbeta = 2
	s = 0.01
	b = 0
	gamma = 0.1
	k  = KernelFunction(gamma = gamma)

	Emb = FaberSchauderEmbedding(d, m, offset=0.2, s = s, b = b,B = 1000., kernel_object=k)
	GP = GaussianProcess(d = d, s = s)
	xtest = torch.from_numpy(interval(n,d))

	x = torch.from_numpy(np.random.uniform(-1,1,N)).view(-1,1)

	F_true = lambda x: torch.sin(x)**2-0.1
	F = lambda x: F_true(x) + s*torch.randn(x.size()[0]).view(-1,1).double()
	y = F(x)
	Emb.fit_gp(x,y)
	GP.fit_gp(x,y)
	mu = Emb.mean_std(xtest)
	mu_true,_ = GP.mean_std(xtest)
	plt.plot(xtest,F_true(xtest),'b', label = 'true')
	plt.plot(xtest,mu_true,'b--', label = 'GP')
	plt.plot(x,y,'ro')
	plt.plot(xtest, mu, 'g-', label = 'positive basis ')
	plt.legend()
	plt.show()
