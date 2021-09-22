from stpy.random_process import RandomProcess
import torch
import numpy as np
import matplotlib.pyplot as plt

class PrimalMKL(RandomProcess):

	def __init__(self,embeddings,init_weights = None, lam = 0.0, s = 0):
		self.embeddings = embeddings
		self.init_weights = init_weights
		self.no_models = len(embeddings)
		self.s = s
		self.lam = lam
		if not isinstance(self.lam,list):
			self.lam = [lam for i in range(self.no_models)]

	def total_embed_dim(self):
		self.dims = []
		for embedding in self.embeddings:
			self.dims.append(embedding.get_basis_size())
		sum = torch.sum(torch.Tensor(self.dims))
		return sum

	def get_emebed_dims(self):
		self.total_embed_dim()
		return self.dims

	# def fit_gp(self, x, y):
	# 	"""
	# 		In this function we are fitting
	# 		In this function we are fitting
	#
	#
	#
	# 	:param x:
	# 	:param y:
	# 	:return:
	# 	"""
	#
	#
	# 	self.x = x
	# 	self.y = y
	# 	(self.n, self.d) = self.x.size()
	# 	self.total_m = self.total_embed_dim()
	# 	dims_index = torch.cumsum(torch.Tensor([0] + self.get_emebed_dims()),dim = 0).int()
	# 	self.w = [torch.ones(size = (i,1), dtype = torch.float64,requires_grad = True)  for i in self.get_emebed_dims()]
	#
	# 	self.theta = torch.ones(size = (self.no_models,1), dtype = torch.float64,requires_grad = True)
	#
	#


	def fit_gp(self,x,y):
		self.x = x
		self.y = y
		(self.n,self.d) = self.x.size()
		self.total_m = self.total_embed_dim()
		dims_index = torch.cumsum(torch.Tensor([0] + self.get_emebed_dims()),dim = 0).int()

		self.w = [torch.ones(size = (i,1), dtype = torch.float64,requires_grad = True)  for i in self.get_emebed_dims()]

		self.theta = torch.ones(size = (self.no_models,1), dtype = torch.float64,requires_grad = True)

		# def cost(theta,w):
		#
		# 	Phi = torch.zeros(size = (self.n,int(self.total_m)), dtype = torch.float64)
		# 	reg = 0.0
		# 	for index,embedding in enumerate(self.embeddings):
		# 		Phi[:,dims_index[index]:dims_index[index+1]] = embedding.embed(self.x)*torch.sqrt(theta[index])
		# 		reg = reg + torch.sqrt(torch.sum((torch.sqrt(theta[index])*w[index])**2))
		# 	wvector = torch.cat(w, 0)
		# 	cost = torch.sum((torch.mm(Phi,wvector) - self.y)**2)
		# 	cost = cost + self.lam*reg
		# 	return cost

		def regularizers(w):
			reg = torch.zeros(self.no_models,dtype=torch.float64)
			for index, embedding in enumerate(self.embeddings):
				reg[index] = torch.sqrt(torch.sum(w[index] ** 2))
			return reg

		def cost(w):
			Phi = torch.zeros(size = (self.n,int(self.total_m)), dtype = torch.float64)
			reg = 0.0
			for index,embedding in enumerate(self.embeddings):
				Phi[:,dims_index[index]:dims_index[index+1]] = embedding.embed_internal(self.x)
				reg = reg + self.lam[index]*torch.sqrt(torch.sum(w[index])**2)
			wvector = torch.cat(w, 0)
			cost = torch.sum((torch.mm(Phi,wvector) - self.y)**2)
			cost = cost + reg**2 + self.s*torch.norm(wvector)**2
			return cost



		## optimizer objective
		loss = torch.zeros(1,1,requires_grad = True,dtype = torch.float64)
		loss = loss + cost(self.w)




		#loss.requires_grad_(True)


		from pymanopt.manifolds import Euclidean, Product
		from pymanopt import Problem
		from pymanopt.solvers import ConjugateGradient
		from stpy.cost_functions import CostFunction

		# define cost function
		C = CostFunction(cost, number_args=self.no_models)
		[cost_numpy, egrad_numpy, ehess_numpy] = C.define()
		x = [np.ones(shape = (i,1))  for i in self.get_emebed_dims()]



		# Optimization with Conjugate Gradient Descent
		#print (cost_numpy(x))
		manifold = Product( [Euclidean(i) for i in self.get_emebed_dims()])
		problem = Problem(manifold=manifold, cost=cost_numpy, egrad=egrad_numpy, ehess=ehess_numpy, verbosity=10)
		#solver = SteepestDescent(maxiter=1000, mingradnorm=1e-8, minstepsize=1e-10)
		solver = ConjugateGradient(maxiter=1000, mingradnorm=1e-8, minstepsize=1e-20)
		Xopt = solver.solve(problem, x=x)





		self.w = [torch.from_numpy(w) for w in Xopt]
		self.theta =  torch.sum(regularizers(self.w),dim = 0)/regularizers(self.w) + self.s
		self.theta = 1./self.theta

		print (self.theta)


	def mean_var(self,xtest):
		n = xtest.size()[0]
		dims_index = torch.cumsum(torch.Tensor([0] + self.get_emebed_dims()),dim = 0).int()
		Phi = torch.zeros(size=(n, int(self.total_m)), dtype=torch.float64)

		for index, embedding in enumerate(self.embeddings):
			Phi[:, dims_index[index]:dims_index[index + 1]] = embedding.embed_internal(xtest)

		wvector = torch.cat(self.w, 0)
		mu = torch.mm(Phi, wvector)

		K = (torch.mm(torch.t(Phi),Phi) + self.s * torch.eye(int(self.total_m), dtype=torch.float64))
		temp = torch.t(torch.solve(torch.t(Phi),K)[0])
		var = torch.sqrt(self.s*self.s*torch.einsum('ij,ji->i', (temp, torch.t(Phi) )).view(-1, 1))

		mu = mu.detach()
		var = var.detach()

		return (mu,var)

	def sample(self,xtest, size =1):
		mu, var = self.mean_var(xtest)
		sample = mu + var
		return sample

	def visualize(self,xtest,f_true = None, points = True, show = True):
		super().visualize(xtest,f_true = f_true, points = points, show = False)
		## histogram of weights
		plt.figure(2)
		plt.bar(np.arange(len(self.embeddings)), self.theta.detach().numpy().flatten(), np.ones(len(self.embeddings)) * 0.5)
		plt.show()


if __name__ == "__main__":
	from stpy.continuous_processes.fourier_fea import GaussianProcessFF
	from stpy.continuous_processes.gauss_procc import GaussianProcess
	from stpy.test_functions.benchmarks import MultiRKHS

	n = 1024
	N = 100
	s = 0.01
	TestFunction = MultiRKHS()
	xtest = TestFunction.interval(n)
	x = TestFunction.initial_guess(N)
	y = TestFunction.eval(x,sigma = s)
	#TestFunction.visualize(xtest)


	GP1 = GaussianProcess(s=0, kernel="linear")
	GP2 = GaussianProcessFF(s=s, m=100, approx="hermite")

	MKL = PrimalMKL([GP1,GP2], lam=[0.1, 0.1], s = s)
	MKL.fit_gp(x, y)

	print ("Importance Weights:",MKL.theta)

	print("Slope of linear line:", MKL.w[0])

	MKL.visualize(xtest, f_true=TestFunction.eval_noiseless)

	# MKL = PrimalMKL(GPs, lam=0.01)
	# MKL.fit_gp(x,y)
	# MKL.visualize(xtest,f_true=TestFunction.eval_noiseless)
	#
	# MKL = PrimalMKL(GPs, lam=0.0001)
	# MKL.fit_gp(x,y)
	# MKL.visualize(xtest,f_true=TestFunction.eval_noiseless)
