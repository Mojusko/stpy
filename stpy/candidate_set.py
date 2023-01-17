import numpy as np
import torch


class CandidateSet():

	def __init__(self):
		pass

class CandidateDiscreteSet(CandidateSet):

	def __init__(self, xtest):
		super().__init__()
		self.xtest = xtest
		self.embedded = False

	def get_set_size(self):
		return self.xtest.size()[0]

	def get_dim(self):
		return self.xtest.size()[1]

	def get_emb_dim(self):
		if self.embedded:
			return self.emb_xtest.size()[1]
		else:
			return self.xtest.size()[1]

	def get_random_elements(self, size = 1):
		n = self.get_set_size()
		indices = np.random.choice(np.arange(0,n,1), size)
		print (indices)
		if self.embedded:
			elem = self.emb_xtest[indices, :]
		else:
			elem = self.xtest[indices,:]
		print (elem)
		return elem

	def debug_subsample(self):
		self.xtest = self.xtest[0:20000,:]

	def get_options_per_dim(self):
		d = {}
		dims = self.get_dim()
		for i in range(dims):
			d[i] = torch.unique(self.xtest[:,i])
		return d

	def get_options(self):
		if self.embedded:
			return self.emb_xtest
		else:
			return self.xtest

	def get_options_raw(self):
		return self.xtest

	def use_embedding(self, embed):
		self.embedded = True
		self.emb_xtest = embed(self.xtest)
