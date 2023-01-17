import numpy as np
import torch

from stpy.point_processes.poisson import PoissonPointProcess


class SeasonalPoissonPointProcess(PoissonPointProcess):

	def __init__(self, *args, seasonality=lambda t: 1., **kwargs):
		self.seasonality = seasonality

	def rate_default(self, x, t, dt=1.):
		return (self.B * torch.sum(torch.exp(-(x + 1)) * torch.sin(2 * x * np.pi) ** 2, dim=1).view(-1,
																									1) + self.b) * dt

	def rate_volume(self, S, t, dt=1, rate=None):
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
				integral, _ = integrate.quad(lambda x: rate(torch.Tensor([x]).view(1, 1), t).numpy(),
											 float(S.bounds[0, 0]), float(S.bounds[0, 1]))
			elif self.d == 2:
				integrand = lambda x, y: rate(torch.Tensor([x, y], t).view(1, 2).double()).numpy()
				integral, _ = integrate.dblquad(integrand, float(S.bounds[0, 0]), float(S.bounds[0, 1]),
												lambda x: float(S.bounds[1, 0]), lambda x: float(S.bounds[1, 1]))

			return integral * dt
		else:
			return self.rate_volume_f(S) * dt

	def sample(self, S, t, dt=1., verbose=False, rate=None):
		"""

		:param S: set where it should be sampled
		:return:
		"""
		if self.exact == True:
			return self.sample_discretized(S, t, dt=dt)
		else:

			lam = self.rate_volume(S, t, dt)
			n = np.random.poisson(lam=lam)
			new_sample = []
			vol = S.volume()
			size = 0

			alpha = 1. / lam

			while size < n:
				# uniform sample g(s) = 1/vol(S)
				sample = S.uniform_sample(1)

				t = self.rate(sample, t) / (alpha * lam)
				p = np.random.uniform(0, 1)
				if p < t:
					new_sample.append(sample.view(1, -1))
					size = size + 1

			if len(new_sample) > 1:
				x = torch.cat(new_sample, dim=0)
			else:
				return None
			return x

	def sample_discretized(self, S, t, dt, n=50):
		lam = float(self.rate_volume(S, t, dt))
		count = np.random.poisson(lam=lam)
		if count > 0:
			x = S.return_discretization(n)
			r = self.rate(x, t) * dt
			sample = torch.from_numpy(
				np.random.choice(np.arange(0, x.size()[0], 1), size=count, p=(r / torch.sum(r)).numpy().reshape(-1)))
			return x[sample, :]
		else:
			return None
