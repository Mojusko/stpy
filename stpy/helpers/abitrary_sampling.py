from typing import Union, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def sample_uniform_sphere(n, d, radius=1):
	X = np.random.randn(n, d)
	X_n = np.random.randn(n, d)
	for i in range(n):
		X_n[i, :] = (X[i, :] / np.linalg.norm(X[i, :])) * radius
	return X_n


def rejection_sampling(pdf, size=(1, 1)):
	"""
	Implements rejection sampling

	:param pdf:
	:param size:
	:return:
	"""
	n = size[0]
	d = size[1]
	output = np.zeros(shape=size)
	i = 0
	while i < n:
		Z = np.random.normal(size=(1, d))
		u = np.random.uniform()
		if pdf(Z) < u:
			output[i, :] = Z
			i = i + 1

	return output


def next_prime():
	def is_prime(num):
		"Checks if num is a prime value"
		for i in range(2, int(num ** 0.5) + 1):
			if (num % i) == 0: return False
		return True

	prime = 3
	while (1):
		if is_prime(prime):
			yield prime
		prime += 2


def vdc(n, base=2):
	vdc, denom = 0, 1
	while n:
		denom *= base
		n, remainder = divmod(n, base)
		vdc += remainder / float(denom)
	return vdc


def halton_sequence(size, dim):
	seq = []
	primeGen = next_prime()
	next(primeGen)
	for d in range(dim):
		base = next(primeGen)
		seq.append([vdc(i, base) for i in range(size)])
	return seq


def sample_qmc_halton_normal(size=(1, 1)):
	Z = np.array(halton_sequence(size[0], size[1])).T
	Z[0, :] += 10e-5
	from scipy.stats import norm
	Z = norm.ppf(Z)
	return Z


def sample_qmc_halton(sampler, size=(1, 1)):
	Z = np.array(halton_sequence(size[0], size[1]), dtype=np.float64).T
	Z[0, :] += 10e-5
	Z = sampler(Z)
	return Z


def sample_bounded(bounds):
	d = len(bounds)
	x = np.zeros(shape=(d))
	for i in range(d):
		x[i] = np.uniform(bounds[i][0], bounds[i][1])
	return x


def randomly_split_set_without_duplicates_balanced(x: torch.Tensor,
											y: torch.Tensor,
											max_bins: int = 2,
										  	alpha: float = 0.2,
										  	size: Union[int, float, None] = None):
	# sort tensor
	N = x.size()[0]

	out, indices = torch.unique(x, dim=0, return_inverse=True)
	n, d = out.size()
	if size is None:
		ntest = int(alpha * n)
	else:
		ntest = size
	y_out = y[np.unique(indices)]

	# bin the data
	samples_per_bin, bins, = np.histogram(y_out, bins=max_bins)  # Doane's method worked best for me
	classes = np.digitize(y_out, bins)
	classes[classes == max_bins+1] = max_bins

	# randomly split
	s = StratifiedShuffleSplit(n_splits=1, test_size=ntest)

	for _, n_test_indices in s.split(out,classes):
		mask_test = torch.zeros(N).bool()
		for index in n_test_indices:
			mask_test = torch.logical_or(mask_test, indices == index)

		return mask_test, ~mask_test


def randomly_split_set_without_duplicates(x: torch.Tensor,
										  alpha: float = 0.2,
										  size: Union[int, float, None] = None):
	"""
	Randomly splits the dataset and returns the mask of the
	:param x:
	:param alpha:
	:return:
	"""

	# sort tensor
	N = x.size()[0]

	out, indices = torch.unique(x, dim=0, return_inverse=True)

	n, d = out.size()
	if size is None:
		ntest = int(alpha * n)
	else:
		ntest = size

	# randomly split
	n_test_indices = np.random.choice(np.arange(0, n, 1), size=ntest, replace=False)
	mask_test = torch.zeros(N).bool()

	for index in n_test_indices:
		mask_test = torch.logical_or(mask_test, indices == index)

	return mask_test, ~mask_test


def randomly_split_set_without_duplicates_general(x: torch.Tensor,
										  sizes: List = [None]):
	"""
	Randomly splits the dataset and returns the mask of the
	:param x:
	:param alpha:
	:return:
	"""

	# sort tensor
	N = x.size()[0]

	out, indices = torch.unique(x, dim=0, return_inverse=True)
	# is number of unique elements
	n, d = out.size()

	# randomly permute indices
	inde = torch.from_numpy(np.random.permutation(np.arange(0, n, 1)))
	cumsum_indices = torch.cumsum(torch.Tensor(sizes),0).int()
	cumsum_indices = torch.cat((torch.Tensor([0]),cumsum_indices)).int()

	masks = [torch.zeros(N).bool() for _ in sizes]
	for j in range(len(sizes)):
		n_test_indices = inde[cumsum_indices[j]:min(n,cumsum_indices[j+1])]
		for index in n_test_indices:
			masks[j] = torch.logical_or(masks[j], indices == index)

	return masks


#


if __name__ == "__main__":
	# x = torch.Tensor([[2, 1, 1], [2, 1, 1], [2, 2, 2],
	# 				  [3, 2, 2], [2, 1, 1], [4, 2, 1],
	# 				  [4, 2, 4], [4,4,4], [1,2,2]]).double()
	#
	x = torch.randint(0, 10, size = (2000,3))
	y = torch.randn(size = (x.size()[0],1))*10

	# masks = randomly_split_set_without_duplicates_general(x, sizes=[1,2,3])
	#
	# for mask in masks:
	# 	print (mask)

	masks = randomly_split_set_without_duplicates_balanced(x,y, size = 100, max_bins = 10)
	masks2 = randomly_split_set_without_duplicates(x, size = 100)
	import matplotlib.pyplot as plt
	labels = ['test', 'train']
	for index,(mask,mask2) in enumerate(zip(masks,masks2)):
		plt.hist(y[mask].T, alpha = 0.2, density= True, label = labels[index])
		plt.hist(y[mask2].T, alpha=0.2, density=True, label=labels[index]+"_random")
	plt.legend()
	plt.show()

