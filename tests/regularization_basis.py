import torch
import numpy as np
import cvxpy as cp

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.helpers.helper import interval
import matplotlib.pyplot as plt
from stpy.kernels import KernelFunction
from stpy.embeddings.bernstein_embedding import BernsteinEmbedding, BernsteinSplinesEmbedding, BernsteinSplinesOverlapping
from stpy.embeddings.bump_bases import TriangleEmbedding,PositiveNystromEmbeddingBump

if __name__ == "__main__":

	d = 1
	m = 32
	n = 256
	N = 20

	s = 0.01
	b = 0.1
	B = 0.5

	gamma = 0.1
	kernel_object = KernelFunction(gamma = gamma)
	kernel_object_poly = KernelFunction(kernel_name="polynomial", power = N)

	EmbBern = BernsteinEmbedding(d,m,kernel_object=kernel_object,offset=0.5,b=b,B=B,s = s)
	EmbSplines = BernsteinSplinesEmbedding(d,m,kernel_object=kernel_object,offset=0.5,b=b,B=B,s = s)
	EmbSplinesOverlap = BernsteinSplinesOverlapping(d,m,kernel_object=kernel_object,offset=0.5,b=b,B=B,s = s)
	Emb = TriangleEmbedding(d,m,kernel_object=kernel_object,offset=0.5,b=b,B=B,s = s)
	Embpoly = TriangleEmbedding(d,m,kernel_object=kernel_object_poly,offset=0.5,b=b,B=B,s = s)
	Embnys = PositiveNystromEmbeddingBump(d, m, kernel_object=kernel_object, offset=0.5, b=0, B=1000, s = s)

	GP = GaussianProcess(d = d, s = s, kernel=kernel_object)

	xtest = torch.from_numpy(interval(n,d,L_infinity_ball=1.1))
	x = torch.from_numpy(np.random.uniform(-1,1,N)).view(-1,1)

	F_true = lambda x: torch.sin(5*x)**2-0.1
	F = lambda x: F_true(x) + s*torch.randn(x.size()[0]).view(-1,1).double()
	y = F(x)

	Emb.fit(x, y)
	EmbBern.fit(x, y)
	Embpoly.fit(x, y)
	EmbSplines.fit(x, y)
	EmbSplinesOverlap.fit(x, y)
	Embnys.fit(x, y)
	GP.fit_gp(x,y)

	mu = Emb.mean_std(xtest)
	mu_spline = EmbSplines.mean_std(xtest)
	mu_spline_overlap = EmbSplinesOverlap.mean_std(xtest)
	mu_true,_ = GP.mean_std(xtest)
	mu_bern = EmbBern.mean_std(xtest)
	mu_poly = Embpoly.mean_std(xtest)
	mu_pos = Embnys.mean_std(xtest)

	plt.plot(xtest, xtest*0+b, 'k--')
	plt.plot(xtest, xtest * 0 + B, 'k--')

	plt.plot(xtest,F_true(xtest),'r', label = 'true')
	plt.plot(xtest,mu_true,'b--', label = 'no-constraints')
	plt.plot(xtest,mu_pos)
	plt.plot(x,y,'ro')
	plt.plot(xtest, mu, 'g-x', label = 'Triangles')
	#plt.plot(xtest, mu_bern, 'y-o',label = 'Bernstein basis')
	#plt.plot(xtest, mu_poly, color = 'orange', label='triangles polynomial kernel')
	#plt.plot(xtest, mu_spline, color='purple', label='splines')
	#plt.plot(xtest, mu_spline_overlap, color='brown', label='splines_overlap')
	plt.legend()
	plt.show()