import torch
import numpy as np
import mosek
from stpy.helpers.helper import interval, cartesian, symsqrt
import matplotlib.pyplot as plt
from stpy.kernels import KernelFunction
import cvxpy as cp
from stpy.random_process import RandomProcess
from stpy.borel_set import BorelSet
import scipy
from scipy.interpolate import BPoly

class PositiveEmbedding(RandomProcess):

	def __init__(self, d,m, kernel_object = None, interval = (-1,1), B=1, b =0, s = 0.001, offset = 0.):
		self.d = d
		self.m = m
		self.b = b
		self.size = self.get_m()
		self.interval = interval
		if kernel_object is None:
			self.kernel_object = KernelFunction()
			self.kernel = lambda x,y: self.kernel_object.kernel(x,y)
		else:
			self.kernel_object = kernel_object
			self.kernel = self.kernel_object.kernel
		self.B = B
		self.s = s
		self.offset = offset

		self.interval = (self.interval[0] - offset,  self.interval[1] + offset)

		self.borel_set = BorelSet(d = 1, bounds = torch.Tensor([[self.interval[0],self.interval[1]]]).double())
		self.mu = None
		self.precomp = False
		self.procomp_integrals = {}
	def integral(self,S):
		pass

	def basis_fun(self,x,j):
		pass

	def get_constraints(self):
		s = self.m**self.d
		l = np.full(s, self.b)
		u = np.full(s, self.B)
		Lambda = np.identity(s)
		return (l,Lambda,u)

	def cov(self, inverse = False):
		if self.precomp == False:
			dm = (self.interval[1] - self.interval[0]) / (self.m - 1)  # delta m
			t = self.interval[0] + torch.linspace(0, self.m - 1, self.m) * dm

			if self.d == 1:
				t = t.view(-1, 1).double()
			elif self.d == 2:
				t = torch.from_numpy(cartesian([t.numpy(), t.numpy()])).double()
			elif self.d == 3:
				t = torch.from_numpy(cartesian([t.numpy(), t.numpy(), t.numpy()])).double()

			self.Gamma = self.kernel(t, t)
			Z = self.embed_internal(t)

			M = torch.pinverse(Z.T @ Z + (self.s)* torch.eye(self.Gamma.size()[0]))
			self.M = torch.from_numpy(np.real(scipy.linalg.sqrtm(M.numpy())))

			#self.Gamma_half = torch.cholesky(Gamma \
			#	+ self.s * self.s * torch.eye(Gamma.size()[0]).double(), upper = True	)

			self.Gamma_half = torch.from_numpy(np.real(scipy.linalg.sqrtm( self.Gamma.numpy() + (self.s**2) *np.eye(self.Gamma.size()[0]))) )
			self.Gamma_half = self.M @ self.Gamma_half
			self.invGamma_half = torch.pinverse(self.Gamma_half)
			self.precomp = True
		else:
			pass

		if inverse == True:
			return self.Gamma_half,self.invGamma_half
		else:
			return self.Gamma_half

	def embed_internal(self,x):
		if self.d == 1:
			out = torch.zeros(size=(x.size()[0], self.m), dtype=torch.float64)
			for j in range(self.m):
				out[:, j] = self.basis_fun(x, j).view(-1)
			return out

		elif self.d == 2:
			phi_1 = torch.cat([self.basis_fun(x[:, 0].view(-1, 1), j) for j in range(0, self.m)], dim=1)
			phi_2 = torch.cat([self.basis_fun(x[:, 1].view(-1, 1), j) for j in range(0, self.m)], dim=1)
			n = x.size()[0]
			out = []
			for i in range(n):
				out.append(torch.from_numpy(np.kron(phi_1[i, :].numpy(), phi_2[i, :].numpy())).view(1, -1))
			out = torch.cat(out, dim=0)
			return out
		elif self.d == 3:
			phi_1 = torch.cat([self.basis_fun(x[:, 0].view(-1, 1), j) for j in range(0, self.m)], dim=1)
			phi_2 = torch.cat([self.basis_fun(x[:, 1].view(-1, 1), j) for j in range(0, self.m)], dim=1)
			phi_3 = torch.cat([self.basis_fun(x[:, 2].view(-1, 1), j) for j in range(0, self.m)], dim=1)

			n = x.size()[0]
			out = []
			for i in range(n):
				out.append(torch.from_numpy(np.kron(phi_3[i,:],np.kron(phi_1[i, :].numpy(), phi_2[i, :].numpy()))).view(1, -1))
			out = torch.cat(out, dim=0)
			return out


	def fit_gp(self, x, y, already_embeded = False):
		m = self.m**self.d

		l, Lambda, u = self.get_constraints()
		Gamma_half = self.cov()

		if already_embeded == False:
			Phi = self.embed(x).numpy()
		else:
			Phi = x.numpy()

		xi = cp.Variable(m)
		obj = cp.Minimize(  self.s**2*cp.norm2(xi) + cp.sum_squares(Phi @ xi - y.numpy().reshape(-1))  )

		constraints = []
		Lambda  = Lambda @ Gamma_half.numpy()
		if not np.all(l == -np.inf):
			constraints.append(Lambda[l != -np.inf] @ xi >= l[l != -np.inf])
		if not np.all(u == np.inf):
			constraints.append(Lambda[u != np.inf] @ xi <= u[u != np.inf])

		prob = cp.Problem(obj, constraints)
		prob.solve(solver=cp.MOSEK, warm_start=False,
				   verbose=False, mosek_params = {mosek.iparam.intpnt_solve_form:mosek.solveform.dual})

		if prob.status != "optimal":
			raise ValueError('cannot compute the mode')

		mode = xi.value
		self.mode = torch.from_numpy(mode).view(-1,1)
		self.mu = self.mode
		return mode


	def embed(self, x):
		Gamma_half = self.cov()
		return self.embed_internal(x) @ Gamma_half

	def mean_std(self, xtest):
		embeding = self.embed(xtest)
		mean = embeding@self.mu
		return mean

	def sample_theta(self):
		self.mu = torch.randn(size = (self.get_m(),1))
		return self.mu

	def sample(self, xtest, size = 1):
		return self.embed(xtest)@self.sample_theta()

	def get_m(self):
		return self.m**self.d