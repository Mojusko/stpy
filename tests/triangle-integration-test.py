import torch
import numpy as np
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.embeddings.bump_bases import TriangleEmbedding
from stpy.embeddings.packing_embedding import PackingEmbedding
from stpy.helpers.helper import interval
import torch
import matplotlib.pyplot as plt
from stpy.borel_set import BorelSet, HierarchicalBorelSets


if __name__ == "__main__":
	d = 1
	m = 64
	S = BorelSet(1,[-1,1])

	embedding = TriangleEmbedding(d=d, m=m, s = 10e-8)

	levels = 5
	hierarchical_structure = HierarchicalBorelSets(d=1, interval=(-1, 1), levels=levels)
	basic_sets = hierarchical_structure.get_sets_level(hierarchical_structure.levels)

	xtest = hierarchical_structure.top_node.return_discretization(512)

	for set in basic_sets:
		print (set.bounds, set.volume())
		x = torch.linspace(set.bounds[0, 0], set.bounds[0, 1], 2)
		Gamma_half = embedding.cov()
		val = torch.sum(torch.pinverse(Gamma_half)@embedding.integral(set))


		plt.plot(x, x * 0 + float(val)/set.volume(), '-o', color="green", lw=5)
	for i in range(m):
		plt.plot(xtest, embedding.basis_fun(xtest,i), 'k')
	plt.show()

	plt.subplot(1,2,1)
	plt.imshow(embedding.M)
	plt.subplot(1,2,2)
	plt.imshow(embedding.Gamma_half)
	plt.show()
	# m = embed_p.size
	# GP = KernelizedFeatures(embeding=embed_p, m=m, d=d)
	# F = lambda x: torch.sin(x)
	# x = torch.from_numpy(interval(2,d))
	# xtest = torch.from_numpy(interval(1024, d))
	# GP.fit_gp(x, F(x))
	# GP.visualize(xtest, f_true=F, show = False)
	# for j in range(p):
	# 	plt.plot(xtest,embed_p.basis_fun(xtest,j+1))
	# plt.show()