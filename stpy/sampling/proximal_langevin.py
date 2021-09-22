from stpy.sampling.langevin import LangevinSampler
import torch
import numpy as np

def ProximalLangevin(LangevinSampler):

	def sample(self, F, nablaF, HessianF, theta0, prox, steps = 100):
		L = self.calculate(HessianF, theta0)
		eta = 0.5 / (L + 1)
		m = theta0.size()[0]
		theta = theta0
		for k in range(steps):
			w = torch.randn(size=(m, 1)).double()
			theta = (1 - eta) * theta - eta * nablaF(theta) + eta * prox(theta) + np.sqrt(2 * eta) * w
			if self.verbose == True:
				print("Iter:", k, theta.T)
		return prox(theta)


def MirrorLangevin(LangvinSampler):
	pass