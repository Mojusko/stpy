import matplotlib.pyplot as plt
import numpy as np
import torch

from stpy.continuous_processes.fourier_fea import GaussianProcessFF
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.embeddings.polynomial_embedding import PolynomialEmbedding, CustomEmbedding
from stpy.estimator import Estimator
from stpy.test_functions.benchmarks import MultiRKHS


class MKL(Estimator):

	def __init__(self, embeddings, init_weights=None, lam=0.0, s=0.1):
		self.embeddings = embeddings
		self.init_weights = init_weights
		self.no_models = len(embeddings)
		self.s = s
		self.lam = lam
		if self.init_weights is None:
			self.init_weights = torch.ones(self.no_models)
		self.weights = self.init_weights
		if not isinstance(self.lam, list):
			self.lam = [lam for i in range(self.no_models)]

	def get_emebed_dims(self):
		self.dims = []
		for embedding in self.embeddings:
			self.dims.append(embedding.get_basis_size())
		return self.dims

	def total_embed_dim(self):
		sum = np.sum(self.get_emebed_dims())
		return sum

	def fit_gp(self, x, y):

		self.x = x
		self.y = y
		(self.n, self.d) = self.x.size()
		self.total_m = self.total_embed_dim()

		self.Reggr = KernelizedFeatures(embeding=self, m=self.total_m, d=d, s=self.s)
		self.Reggr.fit_gp(x, y)

	# def mean_vector(self):
	# 	theta = torch.zeros(size = (self.total_embed_dim()))
	# 	dims_index = torch.cumsum(torch.Tensor([0] + self.get_emebed_dims()),dim = 0).int()
	# 	for index, emb in enumerate(self.embeddings):
	# 		theta_small = emb.sample_theta()
	# 		theta[dims_index[index]:dims_index[index + 1]] = theta_small.view(-1)
	# 	return theta

	def mean_vector(self):
		return self.Reggr.theta_mean()

	def mean_var(self, xtest):
		# mu_avg = torch.zeros(size = (xtest.size()[0],1),dtype = torch.float64)
		# var_avg = torch.zeros(size = (xtest.size()[0],1),dtype = torch.float64)
		#
		# for index, emb in enumerate(self.embeddings):
		# 	mu,var = emb.mean_var(xtest)
		# 	mu_avg = mu_avg + self.weights[index]*mu
		# 	var_avg = var_avg + self.weights[index]*var
		# return [mu_avg,var_avg]

		return self.Reggr.mean_std(xtest)

	def sample(self, xtest, size=1):
		# sample_avg = torch.zeros(size = (xtest.size()[0],1),dtype = torch.float64)
		#
		# for index, emb in enumerate(self.embeddings):
		# 	sample = emb.sample(xtest, size = size)
		# 	sample_avg = sample_avg + self.weights[index]*sample
		return self.Reggr.sample(xtest, size=size)

	def embed(self, xtest):
		n = xtest.size()[0]
		Phi = torch.zeros(size=(n, int(self.total_embed_dim())), dtype=torch.float64)
		dims_index = torch.cumsum(torch.Tensor([0] + self.get_emebed_dims()), dim=0).int()

		for index, embedding in enumerate(self.embeddings):
			Phi[:, dims_index[index]:dims_index[index + 1]] = embedding.embed_internal(xtest)

		return Phi

	def selector_matrix(self):
		dims = []
		for embedding in self.embeddings:
			dims.append(embedding.get_basis_size())
		total_dim = self.total_embed_dim()
		selector = torch.zeros(size=(int(total_dim), self.no_models), dtype=torch.float64)
		z = 0
		for i in range(len(self.embeddings)):
			selector[z:z + dims[i], i] = 1.0
			z = z + dims[i]
		return torch.t(selector)

	###
	def evaluate_design(self, C, Phi):
		n = Phi.size()[0]

		A = torch.lstsq(torch.t(C), torch.t(Phi))[0]
		B = torch.t(A[0:n, :])

		delta = torch.norm(B @ Phi - C, p=2)  # /torch.norm(B, p = 2) #relative error

		pinv = torch.pinverse(torch.t(Phi) @ Phi)
		W = C @ pinv @ torch.t(C)

		rank = torch.matrix_rank(B)
		lambda_max = torch.symeig(W)[0][-1]  # largest eigenvalue

		upper_bound = lambda_max * (self.s * self.s * 2 + delta)

		return [upper_bound.detach(), rank]

	def acquisiton_function(self, C, Phi, candidates):
		values = []
		ranks = []
		for candidate_point in candidates:
			newPhi = torch.cat((Phi, candidate_point.view(1, -1)))
			values.append(self.evaluate_design(C, newPhi)[0])
			ranks.append(self.evaluate_design(C, newPhi)[1])

		return [torch.Tensor(values), torch.Tensor(ranks)]


if __name__ == "__main__":

	n = 16
	N = 4
	s = 0.00000001
	d = 1
	TestFunction = MultiRKHS()
	xtest = TestFunction.interval(n)
	x = TestFunction.initial_guess(N)
	y = TestFunction.eval(x, sigma=s)
	bounds = TestFunction.bounds()

	p = 2
	embedding2 = PolynomialEmbedding(d, p, groups=None)
	GP1 = KernelizedFeatures(embeding=embedding2, m=embedding2.size, d=d, s=s,
							 groups=None, bounds=bounds)

	map = lambda x: torch.abs(x)
	embedding3 = CustomEmbedding(d, map, 1, groups=None)

	GP2 = KernelizedFeatures(embeding=embedding3, m=embedding3.size, d=d, s=s,
							 groups=None, bounds=bounds)

	m = 2
	gamma = 0.2
	GP3 = GaussianProcessFF(d=d, s=s, m=m, gamma=gamma, bounds=bounds, groups=None)
	GP4 = GaussianProcessFF(d=d, s=s, m=m, gamma=gamma, bounds=bounds, groups=None)

	MKL = MKL([GP1, GP2], s=s)

	C = MKL.selector_matrix()
	Candidates = MKL.embed(xtest)
	eps = 1
	N = 1
	x = TestFunction.initial_guess(N)

	plt.close('all')

	while eps > 10e-3:
		# print (x,eps)
		Phi = MKL.embed(x)
		# print (C.size(), Phi.size())
		print(N, MKL.evaluate_design(C, Phi))
		eps = MKL.evaluate_design(C, Phi)[0]
		# N = N + 1
		score, rank = MKL.acquisiton_function(C, Phi, Candidates)
		score = score + 1. / (rank - 1)
		index_min = torch.argmin(score)
		x_min = xtest[index_min]

		plt.plot(xtest.numpy(), torch.log(score).numpy(), 'g')
		plt.plot(xtest.numpy(), rank.numpy(), 'r--')
		plt.plot(x, x * 0, 'ro')
		plt.plot(xtest[index_min].numpy(), torch.log(score[index_min]).numpy(), 'go')
		plt.show()

		x = torch.cat((x, x_min.view(1, -1)))

	y = TestFunction.eval(x, sigma=s)
	print(x)
	print(y)

	MKL.fit_gp(x, y)
	print("Projection:")
	print("--------------")
	print(C @ MKL.mean_vector())
	print("--------------")

	MKL.visualize(xtest, f_true=TestFunction.eval_noiseless)
	plt.show()
