import numpy as np
import torch

from stpy.embeddings.embedding import Embedding


class OnehotEmbedding(Embedding):

	def __init__(self, p, d):
		self.p = p # max value
		self.d = d # sites
		self.m = p*d

	def get_m(self):
		return self.p*self.d


	def apply(self,x,f):
		return torch.stack([f(x_i) for i, x_i in enumerate(torch.unbind(x, dim=0), 0)], dim=0)

	def embed(self, x):
		n,d = x.size()
		out = torch.zeros(n,self.p*self.d).double()

		f = lambda x: torch.from_numpy(np.array([x[i]+20*i for i in range(self.d)])).int()
		indices = self.apply(x,f).long()
		for i in range(n):
			out[i,indices[i]] = 1.

		return out

if __name__ == "__main__":
	emb = OnehotEmbedding(20,2)
	x = torch.Tensor([[2,3],[4,5],[10,19]])
	print (emb.embed(x))