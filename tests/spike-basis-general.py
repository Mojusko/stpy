from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.embeddings.bump_bases import FaberSchauderEmbedding
from stpy.helpers.helper import interval
import torch
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet
if __name__ == "__main__":
	d = 1
	m = 100
	S = BorelSet(1,[-1,1])

	embed_p = FaberSchauderEmbedding(d=d, m=p)
	print (torch.sum(embed_p.integral(S)))

	m = embed_p.size
	GP = KernelizedFeatures(embeding=embed_p, m=m, d=d)
	F = lambda x: torch.sin(x)
	x = torch.from_numpy(interval(2,d))
	xtest = torch.from_numpy(interval(1024, d))
	GP.fit_gp(x, F(x))
	GP.visualize(xtest, f_true=F, show = False)
	for j in range(p):
		plt.plot(xtest,embed_p.basis_fun(xtest,j+1))
	plt.show()