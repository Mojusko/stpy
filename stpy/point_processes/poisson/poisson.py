import matplotlib.pyplot as plt
import numpy as np
import torch

from stpy.borel_set import BorelSet


class PoissonPointProcess():
	"""
	parametrized by log linear model

	"""

	def __init__(self, d=1, B=1, b=0.2, rate=None, rate_volume=None):
		self.B = B
		self.d = d
		self.b = b
		if rate is None:
			self.rate = self.rate_default
		else:
			self.rate = rate

		self.rate_volume_f = rate_volume
		self.exact = True

	def rate_default(self, x, dt=1.):
		return (self.B * torch.sum(torch.exp(-(x + 1)) * torch.sin(2 * x * np.pi) ** 2, dim=1).view(-1,
																									1) + self.b) * dt

	def rate_volume(self, S, dt=1, rate=None):
		if self.rate_volume_f is None:
			# integrate rate numerically over S
			import scipy.integrate as integrate
			if rate is None:
				rate = self.rate
			else:
				rate = rate
			integral = 0
			if self.d == 1:
				# integrate = S.volume()* self.rate(torch.from_numpy(S.bounds[0,1]).view(1))
				integral, _ = integrate.quad(lambda x: rate(torch.Tensor([x]).view(1, 1)).numpy(),
											 float(S.bounds[0, 0]), float(S.bounds[0, 1]))
			elif self.d == 2:
				integrand = lambda x, y: rate(torch.Tensor([x, y]).view(1, 2).double()).numpy()
				integral, _ = integrate.dblquad(integrand, float(S.bounds[0, 0]), float(S.bounds[0, 1]),
												lambda x: float(S.bounds[1, 0]), lambda x: float(S.bounds[1, 1]))

			return integral * dt
		else:
			return self.rate_volume_f(S) * dt

	def sample_discretized(self, S, dt, n=100):
		lam = np.maximum(float(self.rate_volume(S, dt)), 0)
		count = np.random.poisson(lam=lam)
		if count > 0:
			x = S.return_discretization(n)
			r = self.rate(x) * dt
			r = torch.maximum(r, r * 0)
			sample = torch.from_numpy(
				np.random.choice(np.arange(0, x.size()[0], 1), size=count, p=(r / torch.sum(r)).numpy().reshape(-1)))
			return x[sample, :]
		else:
			return None

	def sample_discretized_direct(self, x, val):
		lam = 1000
		count = np.random.poisson(lam=np.maximum(0, lam))
		if count > 0:
			val = torch.abs(val)
			sample = torch.from_numpy(np.random.choice(np.arange(0, x.size()[0], 1),
													   size=count, p=(val / torch.sum(val)).numpy().reshape(-1)))
			return x[sample, :]
		else:
			return None

	def sample(self, S, dt=1., verbose=False, rate=None):
		"""

		:param S: set where it should be sampled
		:return:
		"""
		if self.exact == True:
			return self.sample_discretized(S, dt=dt)
		else:

			lam = self.rate_volume(S, dt)
			n = np.random.poisson(lam=lam)
			print("Number of events:", n)
			alpha = 1.

			new_sample = []
			size = 0
			while size < n:
				# uniform sample g(s) = 1/vol(S)
				sample = S.uniform_sample(1)
				t = self.rate(sample) / (alpha)
				p = np.random.uniform(0, 1)
				if p < t:
					new_sample.append(sample.view(1, -1))
					size = size + 1

			if len(new_sample) > 1:
				x = torch.cat(new_sample, dim=0)
			else:
				return None
			return x

	def rate_sets(self, Sets, dt=1):
		res = []
		for S in Sets:
			res.append(self.rate_volume(S, dt=dt))
		return res

	def visualize(self, S, samples=2, n=10, dt=1., show=True):
		xtest = S.return_discretization(n)
		rate = self.rate(xtest)

		if self.d == 1:
			plt.plot(xtest, rate, label='rate', lw=3)
			for i in range(samples):

				x = self.sample(S, dt=dt)
				if x is not None:
					n = x.size()[0]
					plt.plot(x, x * 0, 'o', label='sample n=' + str(n))

		elif self.d == 2:
			from scipy.interpolate import griddata
			xx = xtest[:, 0].detach().numpy()
			yy = xtest[:, 1].detach().numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), rate[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			fig, ax = plt.subplots(figsize=(15, 7))
			cs = ax.contourf(grid_x, grid_y, grid_z_mu, label='rate')
			ax.contour(cs, colors='k')

			for i in range(samples):
				x = self.sample(S, dt=dt)
				if x is not None:
					ax.plot(x[:, 0].detach().numpy(), x[:, 1].detach().numpy(), 'o', ms=10, alpha=0.5, label='sample')
			ax.grid(c='k', ls='-', alpha=0.1)
			plt.colorbar(cs)

		plt.legend()
		if show == True:
			plt.show()


if __name__ == "__main__":
	d = 2
	n = 100
	bounds = torch.Tensor([[-1, 1], [-1, 1]]).double()
	D = BorelSet(d, bounds)

	process = PoissonPointProcess(d=d, B=2)
	process.visualize(D, samples=10, n=n, dt=10)
