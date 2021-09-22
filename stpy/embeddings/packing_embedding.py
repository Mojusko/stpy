import torch
import numpy as np
from stpy.helpers.helper import interval, cartesian
import matplotlib.pyplot as plt
from stpy.kernels import KernelFunction
import cvxpy as cp
from stpy.random_process import RandomProcess
from stpy.borel_set import BorelSet
import scipy
#from stpy.embeddings.basis_function import SpikeBasis
from scipy.interpolate import BPoly,PPoly

class PackingEmbedding(RandomProcess):

	def __init__(self, d,m,kernel_object = None, basis = 'spikes', interval = (-1,1), B=1,b =0, s = 0.1, offset = 0.):
		self.d = d
		self.m = m
		self.basis = basis
		self.b = b
		self.size = self.get_m()
		self.interval = interval
		if kernel_object is None:
			kernel_object = KernelFunction()
			self.kernel = lambda x,y: kernel_object.kernel(x,y)
		else:
			self.kernel = kernel_object.kernel
		self.B = B
		self.s = s
		self.offset = offset

		self.interval = (self.interval[0] - offset,  self.interval[1] + offset)

		self.basis = basis

	def integrate_1d(self, a,b,tj,dm):
		if a <= tj - dm and b >= tj + dm:  # contained
			vol = 1. * dm
		elif a >= tj + dm or b <= tj - dm:  # outside
			vol = 0.
		elif a <= tj - dm and b >= tj:  # a out , b inside
			res = 1 - np.abs((b - tj) / dm)
			vol = dm * 0.5 + np.abs(b - tj) * (1 + res) / 2
		elif a <= tj - dm and b <= tj:  # a out, b inside
			res = 1 - np.abs((b - tj) / dm)
			vol = 0.5 * np.abs(b - tj) * res
		elif b >= tj + dm and a <= tj:
			res = 1 - np.abs((a - tj) / dm)
			vol = dm * 0.5 + np.abs(a - tj) * (1 + res) / 2
		elif b >= tj + dm and a >= tj:
			res = 1 - np.abs((a - tj) / dm)
			vol = 0.5 * np.abs(a - tj) * res
		else:  # inside
			resa = 1 - np.abs((a - tj) / dm)
			resb = 1 - np.abs((b - tj) / dm)
			vol = (tj - a) * resa / 2 + (b - tj) * resb / 2
		return vol


	def integral(self, S):
		"""
		Integrate the Phi(x) over S
		:param S: borel set
		:return:
		"""
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


		return psi

	def basis_fun(self, x, j):
		"""
		Return the value of basis function \phi_j(x)

		:param x: double, need to be in the interval
		:param j: integer, index of hat functions, 0 <= j <= m-1
		:return: \phi_j(x)
		"""

		if self.basis == 'spikes':
			dm = (self.interval[1] - self.interval[0]) / (self.m -1)  # delta m
			tj = self.interval[0] + (j -1) * dm
			res = 1 - torch.abs((x - tj) / dm)
			res[res < 0] = 0
			return res

		elif self.basis == 'bernstein':
			lim = [self.interval[0], self.interval[1]]
			c = np.zeros(shape=(self.m, 1))
			c[j-1] = 1
			bp = BPoly(c, lim, extrapolate=True)
			res = torch.from_numpy(bp(x.numpy()))
			return res




	def get_constraints(self):
		if self.d == 2:
			s = self.m**2
		else:
			s = self.m
		l = np.full(s, self.b)
		u = np.full(s, self.B)
		Lambda = np.identity(s)
		return (l,Lambda,u)

	def embed_internal(self, x):
		if self.d == 1:
			out = torch.zeros(size = (x.size()[0],self.m), dtype = torch.float64)
			for j in range(self.m):
				out[:,j] = self.basis_fun(x,j+1).view(-1)
			return out
		elif self.d == 2:
			phi_1 = torch.cat([self.basis_fun(x[:, 0].view(-1,1), j) for j in range(1, self.m + 1)],dim = 1)
			phi_2 = torch.cat([self.basis_fun(x[:, 1].view(-1,1), j) for j in range(1, self.m + 1)],dim = 1)
			n = x.size()[0]
			out = []
			for i in range(n):
				out.append(torch.from_numpy(np.kron(phi_1[i,:].numpy(),phi_2[i,:].numpy())).view(1,-1) )
			out = torch.cat(out, dim = 0)
			return out

	def embed(self, x):
		Gamma = self.cov().double()
		Gamma_half = torch.cholesky(Gamma+self.s*torch.eye(Gamma.size()[0]).double())
		return self.embed_internal(x) @ Gamma_half

	def cov(self):
		dm = (self.interval[1] - self.interval[0]) / (self.m - 1)  # delta m
		t = self.interval[0] + torch.linspace(0, self.m - 1, self.m) * dm
		t = t.view(-1, 1).double()

		if self.d == 1:
			Gamma = self.kernel(t,t)
		else:
			t = torch.from_numpy(cartesian([t.numpy(),t.numpy()])).double()
			Gamma = self.kernel(t, t)
		return Gamma,t


	def fit_gp(self, x, y, already_embeded = False):

		m = self.m**d

		l, Lambda, u = self.get_constraints()
		if already_embeded == False:
			Phi = self.embed_internal(x).numpy()
		else:
			Phi = x.numpy()


		Gamma,t  = self.cov()
		Phi2 = self.embed_internal(t)
		xi = cp.Variable(m)
		invGamma = np.linalg.inv(Gamma+ 0.0001 * np.eye(len(Gamma)))
		Reg = Phi2.T @ invGamma @ Phi2


		obj = cp.Minimize(self.s**2*cp.quad_form(xi, Reg) + cp.sum_squares(Phi @ xi -  y.numpy().reshape(-1)))

		constraints = []
		Lambda  = Lambda
		if not np.all(l == -np.inf):
			constraints.append(Lambda[l != -np.inf] @ xi >= l[l != -np.inf])
		if not np.all(u == np.inf):
			constraints.append(Lambda[u != np.inf] @ xi <= u[u != np.inf])

		prob = cp.Problem(obj, constraints)
		prob.solve()

		if prob.status != "optimal":
			raise ValueError('cannot compute the mode')

		self.mu = torch.from_numpy(xi.value).view(-1,1)
		return self.mu

	def mean_std(self, xtest):
		embeding = self.embed_internal(xtest)
		#yvar = torch.diag(embeding @ self.Sigma @ embeding.T).view(-1,1)
		#ystd = torch.sqrt(yvar)
		mean = embeding@self.mu
		return mean, None

	def get_m(self):
		return self.m**self.d

if __name__ == "__main__":
	from stpy.continuous_processes.gauss_procc import GaussianProcess
	d = 1
	m = 2
	n = 256
	N = 20
	sqrtbeta = 2
	lam = 0.01
	s = 0.1
	b = 0.1
	B = 1
	gamma = 0.1
	kernel_object = KernelFunction(gamma = gamma)

	Emb = PackingEmbedding(d, m, s = lam,kernel_object=kernel_object, offset=0.1,b = b, B = B)
	Emb2 = PackingEmbedding(d, m, s = lam,kernel_object=kernel_object, basis = 'bernstein', offset=0.1, b = b, B = B)
	xtest = torch.from_numpy(interval(n,d))

	x = torch.from_numpy(np.random.uniform(-1,1,N)).view(-1,1)

	F_true = lambda x: 0.1*x**3 + 0.5
	F = lambda x: F_true(x) + s*torch.randn(x.size()[0]).view(-1,1).double()
	y = F(x)
	Emb.fit_gp(x,y)
	Emb2.fit_gp(x,y)

	mu,std = Emb.mean_std(xtest)
	mu_bern,std = Emb2.mean_std(xtest)

	plt.plot(xtest,F_true(xtest),'b')
	plt.plot(x,F(x),'ro')

	plt.plot(xtest, xtest*0+b, 'k--')
	plt.plot(xtest, xtest * 0 + B, 'k--')

	plt.plot(xtest, mu, 'r-', label = 'spikes')
	plt.plot(xtest, mu_bern, 'k-', label = 'bernstein')
	plt.legend()
	#plt.plot(xtest,lcb,'orange')

	#plt.fill_between(xtest.numpy().flat, (mu - sqrtbeta * std).numpy().flat, (mu + sqrtbeta * std).numpy().flat, color="green", alpha = 0.3)

	plt.show()

