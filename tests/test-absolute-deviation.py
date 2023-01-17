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
	GP.fit_gp(x, x**4)

	mu = GP.mean_aboslute_deviation(xtest, B = None)
	mu2 = GP.mean_aboslute_deviation(xtest, B = 0.1)
	mu3 = GP.mean_std(xtest)[0]
	mu4 = GP.mean_constrained(xtest, B = 0.1)
	#GP.visualize(xtest, show = False)

	plt.plot(xtest,mu, "--",label = 'l1 unconstrained', alpha = 0.5)
	plt.plot(xtest, mu2,"--",label =  'l1 constrained', alpha = 0.5)
	plt.plot(xtest, mu3, label = 'l2 unconstrained', alpha = 0.5)
	plt.plot(xtest, mu4,label =  'l2 constrained', alpha = 0.5)
	plt.legend()
	plt.show()