import numpy as np
import torch
import matplotlib.pyplot as plt
from stpy.embeddings.bump_bases import BumpsEmbedding
from stpy.embeddings.bernstein_embedding import BernsteinSplinesEmbedding
from stpy.embeddings.bump_bases import PositiveNystromEmbeddingBump
from stpy.kernels import KernelFunction
from stpy.embeddings.packing_embedding import PackingEmbedding
from stpy.embeddings.bump_bases import FaberSchauderEmbedding
from stpy.helpers.helper import interval

m = 16
B4 = FaberSchauderEmbedding(m = m, d = 1)

plt.figure(figsize = (20,20))
basis = lambda x,j: B4.basis_fun(x,j)
x = torch.from_numpy(np.linspace(-1,1,1024)).view(-1,1)
print (B4.hierarchical_mask())
for j in range(m):
	plt.plot(x,basis(x,j), lw = 6)
	plt.grid(ls = '--', lw = 4)
	plt.xlim((-1,1))

plt.show()
