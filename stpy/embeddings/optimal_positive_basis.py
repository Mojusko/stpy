import pickle

import numpy as np
import scipy
import torch

from stpy.borel_set import BorelSet
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.embeddings.positive_embedding import PositiveEmbedding
from stpy.kernels import KernelFunction


class OptimalPositiveBasis(PositiveEmbedding):

	def __init__(self, *args, samples=300, discretization_size=30, saved=False, **kwargs):
		super().__init__(*args, **kwargs)
		self.samples = np.maximum(samples, self.m)

		B = BorelSet(self.d, torch.Tensor([[self.interval[0], self.interval[1]] for _ in range(self.d)]).double())
		self.discretized_domain = B.return_discretization(discretization_size)

		y = self.discretized_domain[:, 0].view(-1, 1) * 0

		print("Optimal basis with arbitrary dimension, namely d =", self.d)
		print("Starting optimal basis construction, with m =", self.m)
		# self.new_kernel_object = KernelFunction(kernel_name=self.kernel_object.optkernel,
		#										gamma = self.kernel_object.gamma, d = self.kernel_object.d)

		self.new_kernel_object = self.kernel_object
		if saved == True:
			print("Did not load GP object, it needs to loaded")
		else:
			self.GP = NystromFeatures(self.new_kernel_object, m=self.m, approx='positive_svd',
									  samples=self.samples)
			self.GP.fit_gp(self.discretized_domain, y)
			print("Optimal basis constructed.")
			if torch.sum(torch.isnan(self.GP.embed(self.discretized_domain))) > 0:
				print("Failed basis? (zero is good):", torch.sum(torch.isnan(self.GP.embed(self.discretized_domain))))
		self.precomp_integral = {}

	def get_m(self):
		return self.m

	def basis_fun(self, x, j):
		return self.GP.embed(x)[:, j].view(-1, 1)

	def embed_internal(self, x):
		out = torch.zeros(size=(x.size()[0], self.m), dtype=torch.float64)
		for j in range(self.m):
			out[:, j] = self.basis_fun(x, j).view(-1)
		return out

	def save_embedding(self, filename):
		filehandler = open(filename, 'w')
		pickle.dump(self.GP, filehandler)

	def load_embedding(self, filename):
		file_pi2 = open(filename, 'r')
		self.GP = pickle.load(file_pi2)

	def get_constraints(self):
		s = self.get_m()
		l = np.full(s, 0.0).astype(float)
		u = np.full(s, 10e10)
		Lambda = np.identity(s)
		return (l, Lambda, u)

	def integral(self, S):
		assert (S.d == self.d)

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
				psi = torch.sum(torch.diag(weights) @ vals, dim=0)
				Gamma_half = self.cov()
				psi = Gamma_half.T @ psi
				self.precomp_integral[S] = psi
				if torch.sum(torch.isnan(psi)) > 0:
					print("Failed integrals? (0 is good):", torch.sum(torch.isnan(psi)))

			else:
				raise NotImplementedError("Higher dimension not implemented.")
			return psi

	def cov(self, inverse=False):

		if self.precomp == False:

			x = self.discretized_domain
			vals = self.GP.embed(x)
			indices = torch.argmax(vals, dim=0)  # the nodes are the maxima of the bump functions
			t = x[indices]
			print("nodes of functions", t.size())

			self.Gamma = self.kernel(t, t)
			Z = self.embed_internal(t)

			M = torch.pinverse(Z.T @ Z + (self.s) * torch.eye(self.Gamma.size()[0]))
			self.M = torch.from_numpy(np.real(scipy.linalg.sqrtm(M.numpy())))

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
	from scipy.interpolate import griddata

	d = 2
	m = 64
	n = 64
	N = 20
	sqrtbeta = 2
	s = 0.01
	b = 0
	gamma = 0.5
	k = KernelFunction(gamma=gamma, d=2)

	Emb = OptimalPositiveBasis(d, m, offset=0.2, s=s, b=b, discretization_size=n, B=1000., kernel_object=k)

	GP = GaussianProcess(d=d, s=s)
	xtest = torch.from_numpy(interval(n, d))

	x = torch.from_numpy(np.random.uniform(-1, 1, size=(N, d)))

	F_true = lambda x: torch.sum(torch.sin(x) ** 2 - 0.1, dim=1).view(-1, 1)
	F = lambda x: F_true(x) + s * torch.randn(x.size()[0]).view(-1, 1).double()
	y = F(x)

	# Try to plot the basis functions
	msqrt = int(np.sqrt(m))
	fig, axs = plt.subplots(msqrt, msqrt, figsize=(15, 7))
	for i in range(m):
		f_i = Emb.basis_fun(xtest, i)  ## basis function
		xx = xtest[:, 0].numpy()
		yy = xtest[:, 1].numpy()
		ax = axs[int(i // msqrt), (i % msqrt)]
		grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
		grid_z_f = griddata((xx, yy), f_i[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
		cs = ax.contourf(grid_x, grid_y, grid_z_f, levels=10)
		ax.contour(cs, colors='k')
		# cbar = fig.colorbar(cs)
		# if self.x is not None:
		#	ax.scatter(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), c='r', s=100, marker="o")
		ax.grid(c='k', ls='-', alpha=0.1)

	plt.savefig("positive.png")
	plt.show()

	Emb.fit(x, y)
	GP.fit_gp(x, y)

	mu, _ = Emb.mean_std(xtest)
	mu_true, _ = GP.mean_std(xtest)

	Emb.visualize_function(xtest, [F_true, lambda x: GP.mean_std(x)[0], lambda x: Emb.mean_std(x)[0]])
	# Emb.visualize_function(xtest,GP.mean_std)
	# Emb.visualize_function(xtest,Emb.mean_std)

	# plt.plot(xtest,mu_true,'b--', label = 'GP')

	# plt.plot(x,y,'ro')
	# plt.plot(xtest, mu, 'g-', label = 'positive basis ')
	# plt.legend()
	plt.show()
