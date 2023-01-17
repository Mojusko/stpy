import matplotlib.pyplot as plt
import torch

from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.embeddings.embedding import Embedding
from stpy.helpers.helper import interval, batch_jacobian, batch_hessian
from stpy.helpers.helper import interval_torch
from stpy.kernels import KernelFunction


class PackingEmbedding(Embedding):

	def __init__(self, d, m, kernel_object, interval=[-1, 1], n=100, method='svd'):
		self.d = d
		self.m = m
		self.interval = interval
		self.size = self.get_m()
		self.kernel_object = kernel_object

		self.kernel = kernel_object.kernel
		self.n = n
		self.method = method
		self.construct()

	def construct(self):
		xtest = interval_torch(self.n, self.d, offset=[self.interval for _ in range(self.d)])
		y = xtest[:, 0].view(-1, 1) * 0

		self.new_kernel_object = KernelFunction(kernel_name=self.kernel_object.optkernel,
												gamma=self.kernel_object.gamma, d=self.d)
		self.GP = NystromFeatures(self.new_kernel_object, m=self.m, approx=self.method)
		self.GP.fit_gp(xtest, y)

	def basis_fun(self, x, j):
		return self.GP.embed(x)[:, j].view(-1, 1)

	def embed(self, x):
		return self.GP.embed(x)

	def _derivative_1(self, x):
		dphi = batch_jacobian(self.embed, x).transpose(0, 1)
		return dphi

	def _derivative_2(self, x):
		d2phi = batch_hessian(self.embed, x).transpose(0, 1).transpose(0, 2)
		return d2phi

	def derivative_1(self, x):
		if self.kernel_object.optkernel == "squared_exponential":
			xs = self.GP.xs
			M = self.GP.M
			derivative = self.kernel_object.derivative_1(xs, x)
			res = torch.einsum('ij,kil->kjl', M, derivative)
			return res
		else:
			dphi = self._derivative_1(x)
		return dphi

	def derivative_2(self, x):
		if self.kernel_object.optkernel == "squared_exponential":
			xs = self.GP.xs
			M = self.GP.M
			derivative = self.kernel_object.derivative_2(xs, x)
			res = torch.einsum('ij,kilm->kjlm', M, derivative)
			return res
		else:
			d2phi = self._derivative_2(x)
		return d2phi


if __name__ == "__main__":
	from stpy.continuous_processes.kernelized_features import KernelizedFeatures

	d = 1
	m = 200
	n = 128
	N = 10

	lam = 1.

	s = 0.0001
	gamma = 0.1

	xtest = torch.from_numpy(interval(n, d))
	x = torch.from_numpy(interval(N, d))

	kernel_object = KernelFunction(gamma=gamma)
	Emb = PackingEmbedding(d, m, kernel_object=kernel_object, n=256, method='nothing')
	print(Emb.GP.M.size())
	GP = KernelizedFeatures(embedding=Emb, m=m, s=s, lam=lam, d=d)
	y = GP.sample(x) * 0
	y[5, 0] = 0.5

	GP.fit_gp(x, y)
	mu, std = GP.mean_std(xtest)

	der = Emb.derivative_1(xtest)[:, :, 0]
	der_comp = Emb._derivative_1(xtest)[:, :, 0]

	print(torch.norm(der - der_comp))

	der = der @ GP.theta_mean()
	der_comp = der_comp @ GP.theta_mean()

	der2 = Emb.derivative_2(xtest)[:, :, 0, 0]
	der2_comp = Emb._derivative_2(xtest)[:, :, 0, 0]

	print(torch.norm(der2 - der2_comp))

	der2 = der2 @ GP.theta_mean()
	der2_comp = der2_comp @ GP.theta_mean()

	plt.plot(xtest, mu)
	plt.plot(xtest, der)
	plt.plot(xtest, der_comp, '--')
	plt.plot(xtest, der2)
	plt.plot(xtest, der2_comp, '--')
	plt.plot(x, y, 'bo')
	plt.grid()
	plt.show()
