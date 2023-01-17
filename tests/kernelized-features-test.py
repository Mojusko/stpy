from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.embeddings.embedding import HermiteEmbedding
from stpy.helpers.helper import interval
import torch
import numpy as np

if __name__ == "__main__":
	m = 16
	gamma = 1.
	s = 0.0001
	n = 40

	embedding = HermiteEmbedding(m = m, gamma = gamma)
	GP = KernelizedFeatures(embedding=embedding,s = s,m = m)

	x = torch.from_numpy(interval(n,1))
	xtest = torch.from_numpy(interval(2048,1))
	F = lambda x: torch.sin(10*x)
	y = F(x)

	GP.fit_gp(x,y)
	mu, std = GP.mean_std(xtest)
	print (mu.size())
	print (std.size())
	GP.visualize(xtest)

	for _ in range(30):
		x = torch.from_numpy(np.random.uniform(-1,1,1)).view(1,1)
		GP.add_data_point(x,F(x))

	GP.visualize(xtest)
