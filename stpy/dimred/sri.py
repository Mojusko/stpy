import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.cluster import KMeans
from stpy.helpers.helper import sample_uniform_sphere

class SRI():

	def __init__(self):
		"""
		:param X: X values
		:param y: response variables
		:param relative: relative to number of samples
		:param buckets:
		"""

	def standardize(self,X):
		(n,d) = X.size()
		Sigma_x = np.cov(self.X.numpy().T)
		E_x = np.mean(self.X.numpy(), axis=0)
		Sigma_x_half_inv = np.linalg.pinv(scipy.linalg.sqrtm(Sigma_x))
		Z = (X.numpy()-np.outer(np.ones(n),E_x)) @ Sigma_x_half_inv

		return Sigma_x_half_inv, Z


	def slice_kmeans(self,y):
		indices = []
		kmeans = KMeans(n_clusters=self.buckets).fit(y.numpy().reshape(-1, 1))

		for label in range(self.buckets):
			ind = kmeans.labels_ == label
			indices.append(ind)
		return indices

	def fit_sri(self,X,y , buckets = 10):
		self.X = X
		self.y = y
		self.buckets = buckets
		(n,d) = self.X.size()
		Sigma_x_half_inv, Z = self.standardize(self.X)

		if isinstance(self.buckets,int):
			indices = self.slice_kmeans(self.y)

			zs = []
			ns = []
			for ind in indices:
				if np.sum(ind) > 1:
					z = np.mean(Z[ind,:].reshape(-1,d), axis = 0)
					ns.append(np.sum(ind))
					zs.append(z)
			Zn = np.array(zs)
			V = (Zn.T @ np.diag(ns) @ Zn) /self.buckets

		else:
			raise AssertionError("Unknown bucketing rule.")

		lams, eignv = np.linalg.eig(V)
		betas =  Sigma_x_half_inv @ eignv
		return torch.from_numpy(lams), torch.from_numpy(betas)


	def fit_save(self,X,y,buckets = 10):
		self.X = X
		self.y = y
		self.buckets = buckets
		(n, d) = self.X.size()
		Sigma_x_half_inv, Z = self.standardize(self.X)

		if isinstance(self.buckets, int):
			indices = self.slice_kmeans(self.y)

			V = np.zeros(shape = (d,d))
			I = np.eye(d)
			for ind in indices:
				ns = np.sum(ind)
				if ns > 1:
					Covar_slice = np.cov(Z[ind, :].reshape(-1, d).T)
					V = V + ((I - Covar_slice) @ (I - Covar_slice))*(float(ns)/float(n))

		else:
			raise AssertionError("Unknown bucketing rule.")

		lams, eignv = np.linalg.eig(V)
		betas = Sigma_x_half_inv @ eignv
		return torch.from_numpy(lams), torch.from_numpy(betas)


	def gradient_design(self,d,k,nablaF, eps = 10e-4):
		Sigma = torch.eye(d).double() * eps
		x0 = torch.rand(size = (k,d)).double()
		subspace = nablaF(x0)
		Sigma = Sigma + subspace.T @ subspace
		return x0, Sigma, subspace

	def sample_dir(self,n,x0,subspace, eps = 10e-4):
		indices = np.arange(0,x0.size()[0],1)
		choice = np.random.choice(indices,n,replace = True)
		magnitude = np.diag(np.random.randn(n))
		sample = x0.numpy()[choice] + magnitude @ subspace[choice].numpy() + eps*np.random.randn(n,d)
		return torch.from_numpy(sample)


if __name__ == "__main__":

	d = 3
	p = 2

	sigma = 0.
	A = torch.from_numpy(np.random.randn(d,p))
	A = torch.from_numpy(np.eye(d,p))
	print (A)
	# exampel function
	f = lambda x: torch.sum((x@A)**2, dim = 1)  + sigma*torch.randn(x.size()[0], dtype=torch.double)
	f_no_noise = lambda x: torch.sum((x@A)**2, dim = 1)

	nablaF = lambda x: x@A @ A.T


	DimRed = SRI()
	N = 100
	x0, Sigma, subspace = DimRed.gradient_design(d,d,nablaF)
	X0 = DimRed.sample_dir(N,x0,subspace)
	y0 = f(X0)

	plt.scatter(X0[:,0],X0[:,1], c = y0.view(-1))
	plt.show()


	lams, betas = DimRed.fit_sri(X0,y0, buckets=20)

	print (lams/torch.sum(lams))
	print (betas)

	lams2, betas2 = DimRed.fit_save(X0,y0, buckets=20)

	print (lams2/torch.sum(lams2))
	print (betas2)