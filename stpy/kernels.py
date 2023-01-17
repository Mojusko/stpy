import math

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.special import kv
from sklearn.metrics.pairwise import check_pairwise_arrays, manhattan_distances
from stpy.kernel_functions.squared_exponential_kernel import squared_exponential_kernel_diag

class KernelFunction:

	def __init__(self, kernel_function=None, kernel_name="squared_exponential", \
				 freq=None, groups=None, d=1, gamma=1, ard_gamma=None, nu=1.5, kappa=1, map=None, power=2,
				 cov=None, params=None, group=None):

		if kernel_function is not None:
			self.kernel_function = kernel_function
			self.optkernel = "custom"
			self.kappa = kappa
			if params is None:
				self.params = {'kappa': self.kappa}
			else:
				self.params = params
			self.initial_params = self.params

			if group is None:
				self.group = [i for i in range(d)]
			else:
				self.group = group
			self.d = d
		else:
			self.optkernel = kernel_name
			self.gamma = gamma
			if ard_gamma is None:
				self.ard_gamma = torch.ones(d).double()
			else:
				try:
					self.ard_gamma = torch.Tensor([ard_gamma]).double()
				except:
					self.ard_gamma = ard_gamma
			self.power = power
			self.v = nu

			if params is not None:
				self.initial_params = params
			else:
				self.initial_params = {'kappa':kappa}

			if cov is None:
				self.cov = torch.eye(d).double()
			else:
				self.cov = cov

			if group is None:
				self.group = [i for i in range(d)]
			else:
				self.group = group

			self.map = map
			self.groups = groups
			self.kappa = kappa
			self.freq = freq
			self.d = d
			self.add = False

		self.kernel_function_list = [self.get_kernel_internal()]
		self.kernel_diag_function_list = [self.get_kernel_internal(diag = True)]
		self.optkernel_list = [self.optkernel]
		self.params_dict = {'0': self.params}
		self.kernel_items = 1

		self.operations = ["-"]

	def __combine__(self, second_kernel_object):
		self.kernel_function_list = self.kernel_function_list + second_kernel_object.kernel_function_list
		self.optkernel_list = self.optkernel_list + second_kernel_object.optkernel_list
		self.operations = self.operations + second_kernel_object.operations[1:]
		for key, value in second_kernel_object.params_dict.items():
			self.params_dict[str(self.kernel_items)] = value
			self.kernel_items += 1

	def __add__(self, second_kernel_object):
		self.__combine__(second_kernel_object)
		diff = len(set(second_kernel_object.group) - set(self.group))
		self.d += diff
		self.operations.append("+")
		return self

	def __mul__(self, second_kernel_object):
		self.__combine__(second_kernel_object)
		self.operations.append("*")
		return self

	def description(self):
		desc = "Kernel description:"
		for index in range(0, self.kernel_items, 1):
			desc = desc + "\n\n\tkernel: " + self.optkernel_list[index]
			desc = desc + "\n\toperation: " + self.operations[index]
			desc = desc + "\n\t" + "\n\t".join(
				["{0}={1}".format(key, value) for key, value in self.params_dict[str(index)].items()])
		return desc

	def add_groups(self, dict):
		for a in self.params_dict.keys():
			if a not in dict.keys():
				dict[a] = {}
			dict[a]['group'] = self.params_dict[a]['group']
		return dict

	def kernel_diag(self, a,b, **kwargs):
		if len(kwargs) > 0:
			# params_dict = list(kwargs)
			# we need to send
			params_dict = kwargs
			self.add_groups(params_dict)
		else:
			params_dict = self.params_dict

		for i in range(0, len(self.kernel_function_list), 1):
			k = self.kernel_diag_function_list[i]
			if str(i) in params_dict.keys():
				arg = params_dict[str(i)]
			else:
				arg = {}
			if self.operations[i] == "+":
				output = output + k(a, b, **arg)
			elif self.operations[i] == "*":
				output = output * k(a, b, **arg)
			else:
				output = k(a, b, **arg)

		return output

	def kernel(self, a, b, **kwargs):

		if len(kwargs) > 0:
			# params_dict = list(kwargs)
			# we need to send
			params_dict = kwargs
			self.add_groups(params_dict)
		else:
			params_dict = self.params_dict

		for i in range(0, len(self.kernel_function_list), 1):
			k = self.kernel_function_list[i]
			if str(i) in params_dict.keys():
				arg = params_dict[str(i)]
			else:
				arg = {}
			if self.operations[i] == "+":
				output = output + k(a, b, **arg)
			elif self.operations[i] == "*":
				output = output * k(a, b, **arg)
			else:
				output = k(a, b, **arg)

		return output

	def get_param_refs(self):
		return self.params_dict

	def get_kernel(self):
		return self.kernel

	def get_kernel_internal(self, diag = False):

		self.params = {**self.initial_params, 'kappa': self.kappa, 'group': self.group}

		if self.optkernel == "squared_exponential":
			self.params = dict(**self.params, **{'gamma': self.gamma})
			if diag:
				return squared_exponential_kernel_diag
			else:
				return self.squared_exponential_kernel

		elif self.optkernel == "ard" and (self.groups is None):
			self.params = dict(**self.params, **{'ard_gamma': self.ard_gamma})
			if diag:
				return self.ard_kernel
			else:
				return self.ard_kernel_diag


		elif self.optkernel == "linear":
			return self.linear_kernel

		elif self.optkernel == "laplace":
			self.params = dict(**self.params, **{'gamma': self.gamma})
			return self.laplace_kernel

		elif self.optkernel == "modified_matern":
			self.params = dict(**self.params, **{'gamma': self.gamma, 'nu': self.v})
			return self.modified_matern_kernel

		elif self.optkernel == "custom":
			return self.kernel_function

		elif self.optkernel == "tanh":
			return self.tanh_kernel

		elif self.optkernel == 'step':
			return self.step_kernel

		elif self.optkernel == "angsim":
			return self.angsim_kernel

		elif self.optkernel == "matern":
			self.params = dict(**self.params, **{'gamma': self.gamma, 'nu': self.v})
			return self.matern_kernel

		elif self.optkernel == "ard_matern":
			self.params = dict(**self.params, **{'ard_gamma': self.ard_gamma, 'nu': self.v})

			if diag:
				return self.ard_matern_kernel_diag
			else:
				return self.ard_matern_kernel

		elif self.optkernel == "full_covariance_se":
			self.params = dict(**self.params, **{'cov': self.cov})
			return self.covar_kernel

		elif (self.optkernel == "polynomial") and (self.groups is None):
			self.params = dict(**self.params, **{'degree': self.power})
			return self.polynomial_kernel

		elif (self.optkernel == "polynomial") and (self.groups is not None):
			self.params = dict(**self.params, **{'degree': self.power, 'groups': self.groups})
			return self.polynomial_additive_kernel

		elif self.optkernel == "ard" and (self.groups is not None):
			self.params = dict(**self.params, **{'ard_gamma': self.ard_gamma, 'groups': self.groups})
			return self.ard_kernel_additive

		elif self.optkernel == "squared_exponential_per_group" and (self.groups is not None):
			self.params = dict(**self.params, **{'groups': self.groups})
			return self.squared_exponential_per_group_kernel_additive
		
		elif self.optkernel == "ard_per_group" and (self.groups is not None):
			self.params = dict(**self.params, **{'groups': self.groups})
			return self.ard_per_group_kernel_additive

		elif self.optkernel == "gibbs":
			self.params = dict(**self.params, **{'groups': self.groups})
			return self.gibbs_kernel

		elif self.optkernel == "gibbs_custom":
			self.params = dict(**self.params, **{'groups': self.groups})
			return self.gibbs_custom_kernel

		elif self.optkernel == "random_map":
			return self.random_map_kernel

		else:
			raise AssertionError("Kernel not implemented.")

	def embed(self, x):
		if self.optkernel == "linear":
			return x
		else:
			raise AttributeError("This type of kernel does not support a finite dimensional embedding")

	def get_basis_size(self):
		if self.optkernel == "linear":
			return self.d
		else:
			raise AttributeError("This type of kernel does not support a finite dimensional embedding")

	def step_kernel(self, a, b, **kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]

		n, d = a.size()
		m, d = b.size()

		K = torch.zeros(size=(n, m)).double()

		for i in range(n):
			for j in range(m):
				K[i, j] = a[i, :] + b[j, :] - torch.abs(a[i, :] - b[j, :])

		return kappa * K.T

	def linear_kernel(self, a, b, **kwargs):
		"""
			GP linear kernel
		"""
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]
		return kappa * (b @ a.T)

	def custom_map_kernel(self, a, b, **kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group


		if 'map' in kwargs.keys():
			map = kwargs['map']
		else:
			map = self.map

		a = a[:, group]
		b = b[:, group]

		if map is not None:
			return kappa * self.linear_kernel(torch.t(self.map.map(a)), torch.t(self.map.map(b))).detach()
		else:
			return kappa * self.linear_kernel(a, b)

	def laplace_kernel(self, a, b, **kwargs):
		if 'gamma' in kwargs.keys():
			gamma = kwargs['gamma']
		else:
			gamma = self.gamma

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa
		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]
		K = - manhattan_distances(a, b) / gamma ** 2
		K = np.exp(K)  # exponentiate K in-place
		return kappa * torch.from_numpy(K).T

	def squared_exponential_kernel(self, a, b, **kwargs):
		"""
			GP squared exponential kernel
		"""
		if 'gamma' in kwargs.keys():
			gamma = kwargs['gamma']
		else:
			gamma = self.gamma

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]
		#	print (a.shape, b.shape)
		normx = torch.sum(a ** 2, dim=1).view(-1, 1)
		normy = torch.sum(b ** 2, dim=1).view(-1, 1)

		product = torch.mm(b, torch.t(a))
		# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy
		arg = (-0.5 / (gamma * gamma)) * sqdist
		res = torch.exp(arg)
		return kappa * res

	def gibbs_custom_kernel(self, a, b, **kwargs):
		if 'gamma_fun' in kwargs.keys():
			gamma_fun = kwargs['gamma_fun']
		else:
			raise AttributeError("Missing gamma_fun in Gibbs kernel definition.")

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa
		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]
		#	print (a.shape, b.shape)
		normx = torch.sum(a ** 2, dim=1).view(-1, 1)
		normy = torch.sum(b ** 2, dim=1).view(-1, 1)

		product = torch.mm(b, torch.t(a))
		# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy

		lengthscales = gamma_fun(a, b)

		arg = (-0.5 / lengthscales) * sqdist
		res = torch.exp(arg)
		return kappa * res

	def gibbs_kernel(self, a, b, **kwargs):
		if 'gamma_fun' in kwargs.keys():
			gamma_fun = kwargs['gamma_fun']
		else:
			raise AttributeError("Missing gamma_fun in Gibbs kernel definition.")

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa
		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]
		#	print (a.shape, b.shape)
		normx = torch.sum(a ** 2, dim=1).view(-1, 1)
		normy = torch.sum(b ** 2, dim=1).view(-1, 1)

		product = torch.mm(b, torch.t(a))
		# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy

		lengthscales = (gamma_fun(a) ** 2 + gamma_fun(b).T ** 2)

		print(lengthscales)

		arg = (-0.5 / lengthscales) * sqdist
		res = torch.exp(arg)
		return kappa * res

	def covar_kernel(self, a, b, **kwargs):
		"""
		:param a:
		:param b:
		:param cov: square-root of the covariance matrix
		:return:
		"""

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'cov' in kwargs.keys():
			cov = kwargs['cov']
		else:
			cov = self.cov
		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]
		a = torch.mm(a, cov)
		b = torch.mm(b, cov)

		normx = torch.sum(a ** 2, dim=1).reshape(-1, 1)
		normy = torch.sum(b ** 2, dim=1).reshape(-1, 1)

		product = torch.mm(b, torch.t(a))
		# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy
		arg = - 0.5 * sqdist
		res = torch.exp(arg)
		return kappa * res

	def ard_kernel(self, a, b, **kwargs):

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'ard_gamma' in kwargs.keys():
			gamma = kwargs['ard_gamma']
		else:
			gamma = self.ard_gamma

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]

		D = torch.diag(1. / (gamma[group]))
		a = torch.mm(a, D)
		b = torch.mm(b, D)
		normx = torch.sum(a ** 2, dim=1).reshape(-1, 1)
		normy = torch.sum(b ** 2, dim=1).reshape(-1, 1)

		product = torch.mm(b, torch.t(a))
		# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy
		arg = - 0.5 * sqdist
		res = torch.exp(arg)
		return kappa * res

	def ard_kernel_diag(self, a, b, **kwargs):

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'ard_gamma' in kwargs.keys():
			gamma = kwargs['ard_gamma']
		else:
			gamma = self.ard_gamma

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]

		D = torch.diag(1. / (gamma[group]))
		a = torch.mm(a, D)
		b = torch.mm(b, D)
		normx = torch.sum(a ** 2, dim=1).reshape(-1, 1)
		normy = torch.sum(b ** 2, dim=1).reshape(-1, 1)

		product = torch.mm(b, torch.t(a))
		# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy
		arg = - 0.5 * sqdist
		res = torch.exp(arg)
		return kappa * res



	def ard_per_group_kernel_additive(self,a,b,**kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa
		
		if 'groups' in kwargs.keys():
			groups = kwargs['groups']
		else:
			groups = self.groups

		if 'ard_per_group' in kwargs.keys():
			ard_per_group = kwargs['ard_per_group']
		else:
			raise AssertionError("This kernel requires 'ard_per_group' initial parameters")

		(n, z) = tuple(a.size())
		(q, m) = tuple(b.size())

		r = torch.zeros(size=(q, n), dtype=torch.float64)
		groups_index = 0

		for group_add in groups:
			kwargs['group'] = group_add
			
			size_group = len(group_add)
			# use per group lenghtscale 
			#kwargs['ard_gamma'] = ard_per_group[groups_index:groups_index+size_group]
			gamma = ard_per_group[groups_index:groups_index+size_group]
			groups_index +=size_group

			ax = a[:, group_add]
			bx = b[:, group_add]
			D = torch.diag(1. / (gamma))
			ax = torch.mm(ax, D)
			bx = torch.mm(bx, D)
			normx = torch.sum(ax ** 2, dim=1).reshape(-1, 1)
			normy = torch.sum(bx ** 2, dim=1).reshape(-1, 1)
			product = torch.mm(bx, torch.t(ax))
			# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
			sqdist = -2 * product + torch.t(normx) + normy
			arg = - 0.5 * sqdist
			res = torch.exp(arg)
			r = r + res

		r = r / float(len(groups))
		return kappa*r

	def squared_exponential_per_group_kernel_additive(self,a,b,**kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa
		
		if 'groups' in kwargs.keys():
			groups = kwargs['groups']
		else:
			groups = self.groups

		if 'gamma_per_group' in kwargs.keys():
			gamma_per_group = kwargs['gamma_per_group']
		else:
			raise AssertionError("This kernel requires 'gamma_per_group' initial parameters")

		(n, z) = tuple(a.size())
		(q, m) = tuple(b.size())

		r = torch.zeros(size=(q, n), dtype=torch.float64)

		for group_add, gamma in zip(groups,gamma_per_group):
			kwargs['group'] = group_add
			
			# use per group lenghtscale 
			kwargs['gamma'] = gamma

			r = r + self.squared_exponential_kernel(a, b, **kwargs)

		r = kappa * r / float(len(groups))
		return r

	def ard_kernel_additive(self, a, b, **kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa
		
		if 'groups' in kwargs.keys():
			groups = kwargs['groups']
		else:
			groups = self.groups

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]

		(n, z) = tuple(a.size())
		(q, m) = tuple(b.size())

		r = torch.zeros(size=(q, n), dtype=torch.float64)

		for group_add in groups:
			kwargs['group'] = group_add
			r = r + self.ard_kernel(a, b, **kwargs)

		r = r / float(len(groups))
		return r

	def tanh_kernel(self, a, b, **kwargs):
		"""
			GP squared exponential kernel
		"""
		#	print (a.shape, b.shape)

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]

		X, Y = check_pairwise_arrays(a.numpy(), b.numpy())
		K = manhattan_distances(a.numpy(), b.numpy())
		K = K.T
		eps = 10e-10
		q = 3
		A = (np.tanh(K) ** q) / (eps + K ** q)
		return kappa * torch.from_numpy(A)

	def angsim_kernel(self, a, b, **kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		return kappa * (2. / np.pi) * np.arcsin((a.dot(b)) / (a.norm() * b.norm()))

	def polynomial_kernel(self, a, b, **kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa
		if 'degree' in kwargs.keys():
			power = kwargs['degree']
		else:
			power = self.power
		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]

		K = (torch.mm(b, torch.t(a)) + 1) ** power
		return kappa * K

	def polynomial_additive_kernel(self, a, b, **kwargs):

		if 'groups' in kwargs.keys():
			groups = kwargs['groups']
		else:
			groups = self.groups
		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]

		(n, z) = tuple(a.size())
		(q, m) = tuple(b.size())
		no_groups = float(len(groups))
		r = torch.zeros(size=(q, n), dtype=torch.float64)
		for i, group in enumerate(groups):
			z = self.polynomial_kernel(a[:, group], b[:, group], **kwargs)
			r = r + z
		r = r / no_groups
		return r


	def matern_kernel(self, a, b, **kwargs):
		"""
		:param a: matrices
		:param b: matrices
		:param gamma: smoothness
		:param v: Bessel function type
		:return:
		"""

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'nu' in kwargs.keys():
			v = kwargs['nu']
		else:
			v = self.v

		if 'gamma' in kwargs.keys():
			gamma = kwargs['gamma']
		else:
			gamma = self.gamma

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group].numpy()
		b = b[:, group].numpy()

		dists = cdist(a / gamma, b / gamma, metric='euclidean').T
		if v == 0.5:
			K = np.exp(-dists)
		elif v == 1.5:
			K = dists * math.sqrt(3)
			K = (1. + K) * np.exp(-K)
		elif v == 2.5:
			K = dists * math.sqrt(5)
			K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
		else:  # general case; expensive to evaluate
			K = dists
			K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
			tmp = (math.sqrt(2 * v) * K)
			K.fill((2 ** (1. - v)) / math.gamma(v))
			K *= tmp ** v
			K *= kv(v, tmp)
		return kappa * torch.from_numpy(K)


	def ard_matern_kernel_diag(self, a, b, **kwargs):
		"""
		:param a: matrices
		:param b: matrices
		:param gamma: smoothness
		:param v: Bessel function type
		:return:
		"""
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'nu' in kwargs.keys():
			v = kwargs['nu']
		else:
			v = self.v

		if 'ard_gamma' in kwargs.keys():
			ard_gamma = kwargs['ard_gamma']
		else:
			ard_gamma = self.ard_gamma

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		D = torch.diag(1. / (ard_gamma[group]))
		a = torch.mm(a, D)
		b = torch.mm(b, D)

		a = a[:, group]
		b = b[:, group]

		#dists = torch.cdist(a , b , p = 2).T
		dists = torch.sqrt(torch.sum((a - b)**2))

		if v == 0.5:
			K = torch.exp(-dists)
		elif v == 1.5:
			K = dists * np.sqrt(3)
			K = (1. + K) * torch.exp(-K)
		elif v == 2.5:
			K = dists * np.sqrt(5)
			K = (1. + K + K ** 2 / 3.0) * torch.exp(-K)
		else:  # general case; expensive to evaluate
			K = dists
			K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
			tmp = (np.sqrt(2 * v) * K)
			K.fill((2 ** (1. - v)) / math.gamma(v))
			K *= tmp ** v
			K *= kv(v, tmp)
		return kappa * K

	def ard_matern_kernel(self, a, b, **kwargs):
		"""
		:param a: matrices
		:param b: matrices
		:param gamma: smoothness
		:param v: Bessel function type
		:return:
		"""

		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'nu' in kwargs.keys():
			v = kwargs['nu']
		else:
			v = self.v

		if 'ard_gamma' in kwargs.keys():
			ard_gamma = kwargs['ard_gamma']
		else:
			ard_gamma = self.ard_gamma

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		D = torch.diag(1. / (ard_gamma[group]))
		a = torch.mm(a, D)
		b = torch.mm(b, D)

		a = a[:, group]
		b = b[:, group]

		dists = torch.cdist(a , b , p = 2).T

		if v == 0.5:
			K = torch.exp(-dists)
		elif v == 1.5:
			K = dists * np.sqrt(3)
			K = (1. + K) * torch.exp(-K)
		elif v == 2.5:
			K = dists * np.sqrt(5)
			K = (1. + K + K ** 2 / 3.0) * torch.exp(-K)
		else:  # general case; expensive to evaluate
			K = dists
			K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
			tmp = (np.sqrt(2 * v) * K)
			K.fill((2 ** (1. - v)) / math.gamma(v))
			K *= tmp ** v
			K *= kv(v, tmp)
		return kappa * K

	def modified_matern_kernel(self, X, Y, **kwargs):
		"""
		:param a: matrices
		:param b: matrices
		:param gamma: smoothness
		:param v: Bessel function type
		:return:
		"""
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'nu' in kwargs.keys():
			v = kwargs['nu']
		else:
			v = self.v

		if 'gamma' in kwargs.keys():
			gamma = kwargs['gamma']
		else:
			gamma = self.gamma

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:, group]
		b = b[:, group]

		d = X.size()[1]
		# Z = np.ones(shape = (X.shape[0],Y.shape[0]))
		Z = torch.ones(size=(Y.size()[0], X.size()[0]), dtype=torch.float64)
		for i in range(d):
			a = X[:, i].view(-1, 1)
			b = Y[:, i].view(-1, 1)
			# dists = cdist(a/gamma,b/gamma,metric='cityblock').T
			dists = cdist(a.numpy() / gamma, b.numpy() / gamma, metric='euclidean').T
			# dists = manhattan_distances(a, b).T/ gamma
			dists = torch.from_numpy(dists)
			if v == 1:
				K = torch.exp(-dists)
			elif v == 2:
				K = (1 + dists) * torch.exp(-dists)
			elif v == 3:
				K = (dists ** 2 + 3 * torch.abs(dists) + 3) * torch.exp(-dists) / 3.
			elif v == 4:
				K = (dists ** 3 + 6 * dists ** 2 + 15 * torch.abs(dists) + 15) * torch.exp(-dists) / 15.
			else:
				raise AssertionError("Kernel with nu = " + str(v) + "not implemented.")
			Z = Z * K
		return kappa * Z

	def spectral_kernel(self, a, b):
		if self.freq is not None:
			(n, d) = a.size()
			(m, d) = b.size()
			dist = torch.zeros(size=(n, m), dtype=torch.float64)
			c = 0
			for x in a:
				z = 0
				for y in b:
					dist[c, z] = torch.sum(torch.cos(torch.mm(x.view(1, 1) - y.view(1, 1), self.freq)))
					z = z + 1
				c = c + 1
			N = self.freq.size()[0]
			return torch.t(dist) / N
		else:
			raise AssertionError("No frequencies passed")

	def wiener_kernel(self, a, b):
		"""
			Wiener process kernel
			k(x,y) = min(x,y)
			k(x,y) = \sum_i min(x_i,y_i)
		"""
		(n, d) = a.size()
		(m, d) = b.size()
		dist = torch.zeros(size=(n, m))
		# dist = 0.1*np.eye(max(n,m))[0:m,0:n]
		c = 0
		for x in a:
			z = 0
			for y in b:
				print(x, y)
				dist[c, z] = torch.from_numpy(np.sum(np.min(np.array([x, y]), axis=0)))
				z = z + 1
			c = c + 1

		# print (dist)
		return dist.T

	def derivative_1(self, fixed, x):
		"""

		"""
		d = x.size()[1]
		n = x.size()[0]

		size = fixed.size()[0]

		if self.optkernel == "squared_exponential":
			k_original = self.squared_exponential_kernel(fixed, x)
			second = fixed.unsqueeze(1) - x
			second = second / self.gamma ** 2
			res = self.kappa * torch.einsum('ij,jik->ijk', k_original, second)
		else:
			raise AssertionError("Not implemented for this kernel")

		# result should be (n,d)
		return res

	def derivative_2(self, fixed, x):
		"""

		"""
		d = x.size()[1]
		n = x.size()[0]

		size = fixed.size()[0]

		if self.optkernel == "squared_exponential":
			k_original = self.squared_exponential_kernel(fixed, x)
			second = fixed.unsqueeze(1) - x
			second = second / self.gamma ** 2
			second2 = torch.einsum('ijk,ijl->ijkl', second, second)
			res1 = torch.einsum('ij,jikl->ijkl', k_original, second2)

			ones = torch.zeros(size=(size, n, d, d))
			for j in range(d):
				ones[:, :, j, j] = 1.
			ones = -ones / self.gamma ** 2
			res2 = torch.einsum('ij,jikl->ijkl', k_original, ones)
			res = self.kappa * (res1 + res2)
		# res = self.kappa * res2
		else:
			raise AssertionError("Not implemented for this kernel")

		return res

	def square_dist(self, a, b):
		if (a.shape == b.shape):
			normx = np.sum(a ** 2, axis=1).reshape(-1, 1)
			normy = np.sum(b ** 2, axis=1).reshape(-1, 1)
		else:
			normx = np.sum(a ** 2, axis=1).reshape(-1, 1)
			normy = np.sum(b ** 2, axis=1).reshape(-1, 1)

		product = b.dot(a.T)
		sqdist = np.tile(normx, b.shape[0]).T + np.tile(normy, a.shape[0]) - 2 * product
		return sqdist
