import torch
from stpy.continuous_processes.fourier_fea import GaussianProcessFF
from stpy.embeddings.embedding import QuadratureEmbedding
from stpy.helpers.helper import interval
if __name__ == "__main__":

	m = 128

	def cost_function():
		gamma = 0.2
		torch.manual_seed(245)
		z2 = QuadratureEmbedding(gamma=gamma, m=m, d=2)
		theta2d = torch.randn(m, 1).double()
		F = lambda x: z2.embed_one(x[1, 0:2].view(1,-1)) @ theta2d
		print (torch.norm(theta2d))
		return F

	F = cost_function()
	xtest = torch.from_numpy(interval(50,2))
	ytest = F(xtest)

	GP = GaussianProcessFF(d = 2, groups=[[0,1]], m = torch.Tensor([m,64]), gamma = torch.Tensor([0.2]))
	GP.fit_gp(xtest,ytest)

	GP.visualize_contour(xtest,f_true=F)

