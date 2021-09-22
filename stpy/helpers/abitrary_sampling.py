import numpy as np
import torch
import cvxpy as cp

def sample_uniform_sphere(n,d,radius=1):
	X = np.random.randn(n,d)
	X_n = np.random.randn(n,d)
	for i in range(n):
		X_n[i,:] = (X[i,:]/np.linalg.norm(X[i,:]))*radius
	return X_n



def rejection_sampling(pdf, size = (1,1)):
	"""
	Implements rejection sampling

	:param pdf:
	:param size:
	:return:
	"""
	n = size[0]
	d = size[1]
	from scipy.stats import norm
	output = np.zeros(shape =size)
	i = 0
	while i < n:
		Z = np.random.normal (size = (1,d))
		u = np.random.uniform()
		if pdf(Z) < u:
			output[i,:] = Z
			i=i+1

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


def sample_qmc_halton_normal(size = (1,1)):
	Z = np.array(halton_sequence(size[0],size[1])).T
	Z[0,:] += 10e-5
	from scipy.stats import norm
	Z = norm.ppf(Z)
	return Z

def sample_qmc_halton(sampler, size = (1,1)):
	Z = np.array(halton_sequence(size[0],size[1]), dtype = np.float64).T
	Z[0,:] += 10e-5
	Z = sampler(Z)
	return Z

def sample_bounded(bounds):
	d = len(bounds)
	x = np.zeros(shape = (d))
	for i in range(d):
		x[i] = np.uniform(bounds[i][0],bounds[i][1])
	return x
