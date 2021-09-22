import torch
import numpy as np
from stpy.random_process import RandomProcess
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.kernels import KernelFunction
import cvxpy as cp


class TraceFeatures(KernelizedFeatures):

	def __init__(self,*args, PSD = False,**kwargs):
		super().__init__(*args,**kwargs)
		self.m = int(self.m)
		self.PSD = PSD

	def construct_covariance(self):
		emb = self.emb
		X = torch.flatten(torch.einsum('ij,ik->jki',emb,emb).permute(1,0,2),end_dim = 1)
		V = torch.einsum('ik,jk->ij',X,X)
		#Z = torch.einsum('ij,j->i',X,y.reshape(-1)).reshape(-1,1)
		self.V = V + self.lam*self.s**2 * torch.eye(self.m**2).double()
		#self.A_new,_ = torch.solve(Z,self.V)
		#self.A_new = self.A_new.reshape(self.m,self.m)

	def fit_gp(self,x,y):
		self.n, self.d = x.size()
		self.x = x
		self.y = y

		self.emb = self.embed(x)
		self.construct_covariance()

		emb = self.emb.numpy()
		A = cp.Variable((self.m, self.m), symmetric=True)
		cost = cp.sum_squares\
				   (cp.diag( emb  @ A @ emb.T ) - y.view(-1).numpy())/(self.s**2)  + (self.lam)* cp.norm(A, "fro")

		if self.PSD == True:
			constraints = [A >> 0]
		else:
			constraints = []

		prob = cp.Problem(cp.Minimize(cost), constraints)
		prob.solve(solver = cp.MOSEK, verbose = True)
		self.A = torch.from_numpy(A.value)
		self.fit = True

	def mean_std(self,xtest, std = True):
		emb = self.embed(xtest)
		mu = torch.einsum('ij,jk,ik->i',emb,self.A,emb).view(-1,1)
		if std == True:
			#invV = torch.inverse(self.V)
			X = torch.flatten(torch.einsum('ij,ik->jki', emb, emb), end_dim=1)
			Z,_ = torch.solve(X,self.V)
			#diagonal = self.lam*self.s ** 2 * torch.einsum('ji,jk,ki->i', (X, invV, X)).view(-1, 1)
			diagonal = self.lam*self.s ** 2 * torch.einsum('ij,ij->j',X,Z).view(-1,1)
			return mu, torch.sqrt(diagonal).view(-1,1)
		else:
			return mu

	def band(self,xtest, sqrtbeta = 2., maximization = True):
		emb = self.embed(xtest)
		X = torch.einsum('ij,ik->ijk', emb, emb)
		n = emb.size()[0]
		ucb = torch.zeros(size = (n,1)).double()

		for i in range(n):
			A = cp.Variable((self.m, self.m), symmetric=True)
			cost = cp.trace(A@X[i,:,:])

			Z = torch.cholesky(self.V, upper = True)
			zero = np.zeros(self.m ** 2)
			constraints = [cp.SOC(zero.T @ cp.vec(A) + self.s*sqrtbeta,Z@(cp.vec(A) - cp.vec(self.A.numpy())))]

			if self.PSD == True:
				constraints += [A >> 0]

			if maximization == True:
				prob = cp.Problem(cp.Maximize(cost), constraints)
			else:
				prob = cp.Problem(cp.Minimize(cost), constraints)

			prob.solve(solver = cp.MOSEK, verbose = False)
			ucb[i] = torch.trace(torch.from_numpy(A.value)@X[i,:,:])
		return ucb

	def lcb(self,xtest, sqrtbeta = 2.):
		return self.band(xtest,sqrtbeta = sqrtbeta, maximization = False)

	def ucb(self, xtest, sqrtbeta=2.):
		return self.band(xtest, sqrtbeta=sqrtbeta, maximization=True)


if __name__ == "__main__":
	from stpy.embeddings.embedding import HermiteEmbedding
	import matplotlib.pyplot as plt

	m = 32
	n = 16
	s = 0.01
	N = 5

	func = lambda x: torch.sin(x*np.pi)**2 + 0.5
	x = torch.from_numpy(np.random.uniform(-1,1,size = (N,1)))
	y = func(x)


	embedding = HermiteEmbedding(m = m, gamma = 0.5)
	xtest = torch.from_numpy(np.linspace(-1,1,n)).view(-1,1)

	F = TraceFeatures(s=s, embedding = embedding, m = m, PSD=True)
	F.fit_gp(x,y)

	F.visualize(xtest, f_true= func, size = 0, show = False)

	lcb = F.lcb(xtest)
	ucb = F.ucb(xtest)
	plt.plot(xtest, lcb,'-s', color='lightblue',label = 'lcb')
	plt.plot(xtest,ucb,'-s', color = 'gray', label = 'ucb')
	plt.legend()
	plt.show()
	#
	# mu, std = F.mean_std(xtest)
	# plt.plot(xtest,func(xtest),'r',label = 'true')
	# plt.plot(xtest,mu,'b', label = 'TR')
	#
	# plt.plot(x,y,'ro',label = 'samples')
	# plt.show()