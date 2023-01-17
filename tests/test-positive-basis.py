import numpy as np
import torch
import matplotlib.pyplot as plt
from stpy.embeddings.bump_bases import BumpsEmbedding
from stpy.embeddings.bernstein_embedding import BernsteinSplinesEmbedding
from stpy.embeddings.bump_bases import PositiveNystromEmbeddingBump
from stpy.kernels import KernelFunction
from stpy.embeddings.packing_embedding import PackingEmbedding
from stpy.helpers.helper import interval

m = 32
kernel = KernelFunction(gamma = 0.1,kernel_name="squared_exponential", power = 5)
B4 = PositiveNystromEmbeddingBump(kernel_object=kernel, m = m, d = 1, samples = 100)

plt.figure(figsize = (20,20))
basis = lambda x,j: B4.basis_fun(x,j)
x = torch.from_numpy(np.linspace(-1,1,100)).view(-1,1)

for j in range(m):
	plt.plot(x,basis(x,j), lw = 6)
	plt.grid(ls = '--', lw = 4)
	plt.xlim((-1,1))

plt.show()
