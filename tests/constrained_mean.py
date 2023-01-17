from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.embeddings.polynomial_embedding import ChebyschevEmbedding
from stpy.helpers.helper import interval
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
	d = 1
	p = 4
	embed_p = ChebyschevEmbedding(d=d, p=p)
	m = embed_p.size
	GP = KernelizedFeatures(embeding=embed_p, m=m, d=d)

	x = torch.from_numpy(interval(10,d))
	xtest = torch.from_numpy(interval(1024, d))
	GP.fit_gp(x, x**8)

	mu = GP.mean_constrained(xtest, B = 0.5)

	GP.visualize(xtest, show = False)
	#plt.plot(x, x**8,'o')
	plt.plot(xtest,mu)
	plt.show()