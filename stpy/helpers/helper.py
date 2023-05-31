# import numpy as np
import copy
import itertools
from typing import Union

import numpy as np
import torch
from torch.autograd.functional import jacobian


def isin(element, test_elements, assume_unique=False, atol = 1e-10):
	(n, d) = element.shape
	(m, d) = test_elements.shape
	maskFull = np.full((n), False, dtype=bool)
	for j in range(m):
		mask = np.full((n), True, dtype=bool)
		for i in range(d):
			# mask = np.logical_and(mask,np.in1d(element[:,i],test_elements[j,i], assume_unique=assume_unique))
			mask = np.logical_and(mask, np.isclose(element[:, i], test_elements[j, i], atol=atol))
		# print (j, i, mask)
		maskFull = np.logical_or(mask, maskFull)
	# print (maskFull)
	return maskFull



def cartesian(arrays, out=None, dtype = None):
	"""
	Generate a cartesian product of input arrays.

	Parameters
	----------
	arrays : list of array-like
			1-D arrays to form the cartesian product of.
	out : ndarray
			Array to place the cartesian product in.

	Returns
	-------
	out : ndarray
			2-D array of shape (M, len(arrays)) containing cartesian products
			formed of input arrays.

	"""
	arrays = [np.asarray(x) for x in arrays]
	if dtype is None:
		dtype = arrays[0].dtype
	n = np.prod([x.size for x in arrays])
	if out is None:
		out = np.zeros([n, len(arrays)], dtype=dtype)

	m = n / arrays[0].size
	m = int(m)
	out[:, 0] = np.repeat(arrays[0], m)
	if arrays[1:]:
		cartesian(arrays[1:], out=out[0:m, 1:])
		for j in range(1, arrays[0].size):
			out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
	return out


def estimate_std(x: torch.Tensor, # x values used for uniqueness detection
				 y: torch.Tensor, # y values
				 truncation:Union[float,None] = None, # truncate at specific y
				 verbose:bool = False, # verbosity level
				 conservative:bool = False,
				 return_all_residuals:bool = False # return
				 ): #

	out, indices, counts = torch.unique(x, dim=0, return_inverse=True, return_counts=True)
	residuals_mean_list = []

	for i in range(counts.size()[0]):
		if counts[i] > 1:
			mask = indices == i
			mean = torch.mean(y[mask].view(-1))
			residuals_mean_list.append(y[mask].view(-1)-mean.view(-1))
	residuals_mean = torch.hstack(residuals_mean_list)

	if verbose:
		print ("Estimating variance from:",residuals_mean.size())

	if truncation is not None:
		residuals_mean_trunc = residuals_mean[torch.abs(residuals_mean)<truncation]
		sigma_std = torch.std(residuals_mean_trunc)
	else:
		sigma_std = torch.std(residuals_mean)

	if return_all_residuals:
		return residuals_mean_list, out, counts, residuals_mean, indices
	else:
		return sigma_std



def direct_sum(arrays):
	dim = np.sum([array.shape[1] for array in arrays])
	size = np.sum([array.shape[0] for array in arrays])

	out = np.zeros(shape=(size, dim))
	dim = 0
	n = 0
	for j in range(len(arrays)):
		new_n, new_dim = arrays[j].shape
		out[n:n + new_n, dim:dim + new_dim] = arrays[j]
		dim = dim + new_dim
		n = n + new_n

	return out


def symsqrt(matrix):
	"""Compute the square root of a positive definite matrix."""
	# perform the decomposition
	# s, v = matrix.symeig(eigenvectors=True)
	_, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
	# truncate small components
	above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
	s = s[..., above_cutoff]
	v = v[..., above_cutoff]
	# compose the square root matrix
	return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


def interval(n, d, L_infinity_ball=1, offset=None):
	if offset is None:
		arrays = [np.linspace(-L_infinity_ball, L_infinity_ball, n).reshape(n, 1) for i in range(d)]
		xtest = cartesian(arrays)
	else:
		arrays = [np.linspace(offset[i][0], offset[i][1], n).reshape(n, 1) for i in range(d)]
		xtest = cartesian(arrays)
	return xtest


def interval_torch(n, d, L_infinity_ball=1, offset=None):
	return torch.from_numpy(interval(n, d, L_infinity_ball=L_infinity_ball, offset=offset))


def get_ecdf(x):
	x = np.sort(x)

	def result(v):
		return np.searchsorted(x, v, side='right') / x.size

	return result


def emprical_cdf(data):
	"""
	#>>> import numpy as np
	#>>> emprical_cdf(np.array([1.,2.,3.,1.,2.]))
	#[1.,2.,3.],[0.4,0.4,0.2]
	"""

	# create a sorted series of unique data
	cdfx = np.sort(np.unique(data))
	# x-data for the ECDF: evenly spaced sequence of the uniques
	x_values = np.linspace(start=min(cdfx),
						   stop=max(cdfx), num=len(cdfx))

	# size of the x_values
	size_data = data.shape[0]
	# y-data for the ECDF:
	y_values = []
	for i in x_values:
		# all the values in raw data less than the ith value in x_values
		temp = data[data <= i]
		# fraction of that value with respect to the size of the x_values
		value = float(temp.shape[0]) / float(size_data)
		# pushing the value in the y_values
		y_values.append(value)
	# return both x and y values
	return x_values, np.array(y_values)


def batch_jacobian(f, x, create_graph=False, vectorize=False):
	f_sum = lambda x: torch.sum(f(x), axis=0)
	return jacobian(f_sum, x, create_graph=create_graph, vectorize=vectorize)


def batch_hessian(f, x, create_graph=False, vectorize=False, vv=False):
	J = lambda x: batch_jacobian(f, x, create_graph=True, vectorize=vectorize).transpose(0, 1)
	H = batch_jacobian(J, x, create_graph=create_graph, vectorize=vv)
	return H


def create_pull_back(low, high, inverse=False, to=[-1, 1]):
	translate = lambda x: x * (to[0] - to[1]) / (low - high) + to[1] - ((to[0] - to[1]) * high) / (low - high)
	if inverse:
		translate_back = lambda x: x * (low - high) / (to[0] - to[1]) + high - to[1] * (low - high) / (to[0] - to[1])
		return translate, translate_back
	else:
		return translate


def hierarchical_distance(group1, group2):
	group3 = copy.deepcopy(group2)
	group4 = copy.deepcopy(group1)
	for elem in group1:
		try:
			group3.remove(elem)
			group4.remove(elem)
		except:
			pass
	if len(group3) == 1 and len(group3[0]) == 1 and len(group4) == 0:
		return 1

	isin = lambda set, set2: []
	for a, b in list(itertools.product(group1, group1)):
		new_group = copy.deepcopy(group1)
		if a != b:
			new_group.remove(b)
			new_group.remove(a)
			new_group.append(a + b)
			if len(new_group) == len(group2) and all(i in new_group for i in group2):
				return 1
	return 2


def valid_enlargement(curr, groups):
	out = []
	for index, group in enumerate(groups):
		if hierarchical_distance(curr, group) == 1:
			out.append(index)
	return out


def interval_groups(n, d, groups, L_infinity_ball=1):
	arrays = [interval(n, len(groups[i]), L_infinity_ball=L_infinity_ball) for i in range(len(groups))]
	xtest = direct_sum(arrays)
	out = np.zeros(shape=(xtest.shape[0], d))
	out[:, 0:xtest.shape[1]] = xtest
	return out


def logsumexp(a, axis=None, b=None):
	a = np.asarray(a)
	if axis is None:
		a = a.ravel()
	else:
		a = np.rollaxis(a, axis)
	a_max = a.max(axis=0)
	if b is not None:
		b = np.asarray(b)
		if axis is None:
			b = b.ravel()
		else:
			b = np.rollaxis(b, axis)
		out = np.log(np.sum(b * np.exp(a - a_max), axis=0))
	else:
		out = np.log(np.sum(np.exp(a - a_max), axis=0))
	out += a_max
	return out


class MyBounds(object):
	def __init__(self, xmax=[1.1, 1.1], xmin=[-1.1, -1.1]):
		self.xmax = np.array(xmax)
		self.xmin = np.array(xmin)

	def __call__(self, **kwargs):
		x = kwargs["x_new"]
		tmax = bool(np.all(x <= self.xmax))
		tmin = bool(np.all(x >= self.xmin))
		return tmax and tmin


def full_group(d):
	g = []
	for i in range(d):
		g.append([i])
	return g


def pair_groups(d):
	g = []
	for i in range(d - 1):
		g.append([i, i + 1])
	return g


def conditional_decorator(dec, condition):
	def decorator(func):
		if not condition:
			# Return the function unchanged, not decorated.
			return func
		return dec(func)

	return decorator

def generate_all_pairs(d):
	groups = []
	for elem in range(d):
		for elem2 in range(d):
			groups.append([elem, elem2])
	return groups


def generate_groups(d, elements=None):
	"""
	returns a list of all possible groups combinations of d elements
	:param d: integer
	:return:
	>>> generate_groups(1)
	[[0]]
	>>> generate_groups(2)
	[[[0], [1]], [[1], [0]], [[0, 1]]]
	"""
	if elements is None:
		elements = list(range(d))
	g = []
	if len(elements) == 1:
		return [elements]

	for r in range(1, d + 1, 1):
		gn = [list(a) for a in list(itertools.combinations(elements, r))]
		for i in gn:
			elements2 = list(set(elements) - set(i))
			g.append([i] + generate_groups(d, elements=elements2))
	return g


class results:
	def __init__(self):
		self.x = 0


def proj(x, bounds):
	y = np.zeros(shape=x.shape)
	for ind, elem in enumerate(x):
		if elem > bounds[ind][1]:
			y[ind] = bounds[ind][1]

		elif elem < bounds[ind][0]:
			y[ind] = bounds[ind][0]

		else:
			y[ind] = elem
	return y


def lambda_coordinate(fun, x0, index, x):
	x0[index] = x
	r = fun(x0)
	return r


def projected_gradient_descent(fun, grad, x, bounds, maxit=10e23, verbose=False, tol=0.000001, nu=0.001):
	i = 0
	x_old = x + np.random.randn(x.shape[0])
	while (i < maxit and np.linalg.norm(x - x_old) > tol):
		x_old = x
		x = x - (100 * nu) * grad(x)
		x = proj(x, bounds)

		if verbose == True:
			print("Iteration: ", i, " ", fun(x))
		i += 1
	res = results()
	res.x = x
	return res


def projected_gradient_descent(fun, grad, x, bounds, maxit=10e23, verbose=False, tol=0.000001, nu=0.001):
	i = 0
	x_old = x + np.random.randn(x.shape[0])
	while (i < maxit and np.linalg.norm(x - x_old) > tol):
		x_old = x
		x = x - (100 * nu) * grad(x)
		x = proj(x, bounds)

		if verbose == True:
			print("Iteration: ", i, " ", fun(x))
		i += 1
	res = results()
	res.x = x
	return res


def complex_step_derivative(fun, h, x):
	d = x.shape[1]
	der = np.zeros(shape=(1, d))
	for i in range(d):
		one = np.zeros(shape=(1, d))
		one[0, i] = 1.0
		der[0, i] = np.imag((fun(x + 1j * h * one) - fun(x))) / h
	return der


def finite_differences(fun, h, x):
	d = x.size()[1]
	der = torch.zeros(size=(1, d), dtype=torch.float64)
	for i in range(d):
		one = torch.zeros(size=(1, d), dtype=torch.float64)
		one[0, i] = 1.0
		der[0, i] = (fun(x + one * h) - fun(x)) / h
	return der


def finite_differences_hessian(fun, h, x):
	d = x.size()[1]
	hess = torch.zeros(size=(d, d), dtype=torch.float64)
	for i in range(d):
		for j in range(d):
			one_i = torch.zeros(size=(1, d), dtype=torch.float64)
			one_j = torch.zeros(size=(1, d), dtype=torch.float64)
			one_i[0, i] = 1.0
			one_j[0, j] = 1.0
			hess[i, j] = np.log(
				np.abs(fun(x + h * one_i + h * one_j) - fun(x + h * one_i) - fun(x + h * one_j) + fun(x))) - 2 * np.log(
				h)

	hess = torch.exp(hess)
	return (hess + torch.t(hess)) / 2.


def finite_differences_np(fun, h, x):
	d = x.shape[0]
	der = np.zeros(shape=(d))
	for i in range(d):
		one = np.zeros(shape=(d))
		one[i] = 1.0
		der[i] = (fun(x + one * h) - fun(x)) / h
	return der


def finite_differences_test(fun, fun_der, x, h_max=1.):
	n = 10
	for i in range(n):
		h = 2 ** (-i) * h_max
		approx_nabla = finite_differences_np(fun, h, x)
		print(i, h, np.linalg.norm(approx_nabla - fun_der(x)))


def sample_custom(inverse_cumulative_distribution, size=(1, 1)):
	U = np.random.uniform(0, 1, size=size)
	F = np.vectorize(inverse_cumulative_distribution)
	Z = F(U)
	return Z


def select_subset(M, S):
	d = M.shape[0]
	I = np.zeros(shape=(d, d))
	I[S, S] = 1.
	return I @ M @ I


def select_subset_inv(M, S):
	M = select_subset(M, S)
	return np.linalg.pinv(M)


def complement_set(S, size):
	V = set(np.arange(0, size, 1))
	s = V - set(S)
	S_C = list(s)
	return S_C


def add_element(elements, new_element):
	new_out = []
	for element in elements:
		new_out.append(element + [[new_element]])
		new_out.append(element)
		for j in element:
			new = copy.deepcopy(element)
			new.remove(j)
			new.append(j + [new_element])
			new_out.append(new)

	return new_out


def get_hierarchy(start=1, new_elements=[2, 3, 4]):
	elements = [[[start]]]
	for new_element in new_elements:
		elements = add_element(elements, new_element)
	l = []
	for element in elements:
		l.append(np.sum([3 ** len(e) for e in element]))
	indices = np.argsort(l)
	out = []
	for index in indices:
		out.append(elements[index])
	return out


def likelihood_bernoulli_test(alpha, delta, failure):
	if alpha == 1.:
		alpha = 0.99999

	p = (1 - (np.log(alpha / delta)) / np.log((1 - alpha) / (1 - delta))) ** (-1)

	dkl = p * np.log(p / delta) + (1 - p) * np.log((1 - p) / (1 - delta))
	n = np.log(2 / failure) / dkl
	k = n * p
	return n, k


def median_of_means(list, delta=0.01):
	r = list.shape[0]
	if r > 3:
		k = r
		N = int(np.floor(r / k))
		means = []
		for j in range(k - 1):
			means.append((1. / N) * np.sum(list[(j * N):(j + 1) * N]))
		return np.median(means)
	else:
		return 0.


def get_indices(xtest,x):
	"""
	Find location of vectors in a larger set
	:param xtest: torch.Tensor, tensor to be located
	:param x: torch.Tensor, to be located in xtest
	:return: list, if None its means it was not found in the original tensor
	"""

	indices = []
	for i in range(x.size()[0]):
		xtrial = x[i,:]
		mask = torch.all(xtest == xtrial, dim=1)
		if torch.sum(mask) > 0:
			index = int(torch.argmax(mask.int()))
			indices.append(index)
		else:
			indices.append(None)

	return indices

if __name__ == "__main__":
	x = torch.arange(0,9,1).reshape(3,3)
	xtrial = torch.Tensor([[0,1,2],[6,7,8],[3,4,5]])
	print (x)
	print (get_indices(x,xtrial))
