import torch


class BernoulliPointProcess():

	def __init__(self, basic_sets, d=1, rate=None):
		self.basic_sets = basic_sets
		self.rate = rate
		self.d = d

	def is_basic(self, S):
		"""
		:return:
		"""
		for set in self.basic_sets:
			if hash(set) == hash(S):
				return True
		return False

	def sample(self, S, t=None, dt=None):
		if self.is_basic(S):
			rv = torch.bernoulli(self.rate(S))
			if rv > 0.5:
				return (S, 1., 1., dt, t)
			else:
				return (S, 0., 1., dt, t)
		else:
			# iterate over all sets that contain it
			outcome = 0.
			for set in self.basic_sets:
				if S.inside(set):
					rv = float(torch.bernoulli(self.rate(S)))
					outcome = max(rv, 0.)
				if outcome > 0.5:
					return (S, 1., 1., dt, t)
				else:
					return (S, 0., 1., dt, t)
			pass
