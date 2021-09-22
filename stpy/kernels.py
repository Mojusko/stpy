import numpy as np
import torch
import math
from scipy.special import kv, gamma
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import check_pairwise_arrays, manhattan_distances

class KernelFunction:

	def __init__(self,kernel_function = None, kernel_name = "squared_exponential", \
				 freq = None, groups = None, d = 1, gamma = 1, ard_gamma = None, nu = 1, kappa =1, map = None, power = 2,
				 cov = None, params = None, group = None):

		if kernel_function is not None:
			self.kernel_function = kernel_function
			self.optkernel = "custom"
			self.kappa = kappa
			if params is None:
				self.params = {'kappa':self.kappa}
			else:
				self.params = params

			if group is None:
				self.group = [i for i in range(d)]
			else:
				self.group = group

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
		self.optkernel_list = [self.optkernel]
		self.params_dict = {'0':self.params}
		self.kernel_items = 1

		self.operations = ["-"]

	def __combine__(self, second_kernel_object):
		self.kernel_function_list = self.kernel_function_list + second_kernel_object.kernel_function_list
		self.optkernel_list = self.optkernel_list + second_kernel_object.optkernel_list
		self.operations = self.operations + second_kernel_object.operations[1:]
		for key,value in second_kernel_object.params_dict.items():
			self.params_dict[str(self.kernel_items)] = value
			self.kernel_items +=1

	def __add__(self, second_kernel_object):
		self.__combine__(second_kernel_object)
		diff = len(set(second_kernel_object.group)-set(self.group))
		self.d += diff
		self.operations.append("+")
		return self

	def __mul__(self, second_kernel_object):
		self.__combine__(second_kernel_object)
		self.operations.append("*")
		return self

	def description(self):
		desc = "Kernel description:"
		for index in range(0,self.kernel_items,1):
			desc = desc+"\n\n\tkernel: " + self.optkernel_list[index]
			desc = desc + "\n\toperation: " + self.operations[index]
			desc = desc+"\n\t" + "\n\t".join(["{0}={1}".format(key,value) for key,value in self.params_dict[str(index)].items() ])
		return desc

	def add_groups(self,dict):
		for a in self.params_dict.keys():
			if a not in dict.keys():
				dict[a] = {}
			dict[a]['group'] = self.params_dict[a]['group']
		return dict

	def kernel(self,a,b,**kwargs):

		if len(kwargs) > 0:
			#params_dict = list(kwargs)
			# we need to send
			params_dict = kwargs
			self.add_groups(params_dict)
		else:
			params_dict = self.params_dict

		for i in range(0,len(self.kernel_function_list),1):
			k = self.kernel_function_list[i]
			if str(i) in params_dict.keys():
				arg = params_dict[str(i)]
			else:
				arg = {}
			if self.operations[i] == "+":
				output = output + k(a,b,**arg)
			elif self.operations[i] == "*":
				output = output * k(a,b,**arg)
			else:
				output = k(a,b,**arg)

		return output

	def get_param_refs(self):
		return self.params_dict

	def get_kernel(self):
		return self.kernel

	def get_kernel_internal(self):
		self.params = {'kappa':self.kappa,'group':self.group}

		if self.optkernel == "squared_exponential":
			self.params = dict(**self.params,**{'gamma': self.gamma})
			return self.squared_exponential_kernel

		elif self.optkernel == "ard" and (self.groups is None):
			self.params = dict(**self.params,**{'ard_gamma': self.ard_gamma})
			return self.ard_kernel

		elif self.optkernel == "linear":
			return self.linear_kernel

		elif self.optkernel == "laplace":
			self.params = dict(**self.params,**{'gamma': self.gamma})
			return self.laplace_kernel

		elif self.optkernel == "modified_matern":
			self.params = dict(**self.params,**{'gamma': self.gamma, 'nu':self.v})
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
			self.params = dict(**self.params,**{'gamma': self.gamma, 'nu':self.v})
			return self.matern_kernel

		elif self.optkernel == "full_covariance_se":
			self.params=  dict(**self.params,**{'cov':self.cov})
			return self.covar_kernel

		elif (self.optkernel == "polynomial") and (self.groups is None):
			self.params = dict(**self.params,**{'degree':self.power})
			return self.polynomial_kernel

		elif (self.optkernel == "polynomial") and (self.groups is not None):
			self.params = dict(**self.params,**{'degree':self.power,'groups':self.groups})
			return self.polynomial_additive_kernel

		elif self.optkernel == "ard" and (self.groups is not None):
			self.params = dict(**self.params,**{'ard_gamma': self.ard_gamma, 'groups':self.groups})
			return self.ard_kernel_additive

		elif self.optkernel == "random_map":
			return self.random_map_kernel

		else:
			raise AssertionError("Kernel not implemented.")

	def embed(self,x):
		if self.optkernel == "linear":
			return x
		else:
			raise AttributeError("This type of kernel does not support a finite dimensional embedding")

	def get_basis_size(self):
		if self.optkernel == "linear":
			return self.d
		else:
			raise AttributeError("This type of kernel does not support a finite dimensional embedding")


	def step_kernel(self,a,b, **kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:,group]
		b = b[:,group]

		n, d = a.size()
		m, d = b.size()

		K = torch.zeros(size = (n,m)).double()

		for i in range(n):
			for j in range(m):
				K[i,j] = a[i,:]+b[j,:]-torch.abs(a[i,:]-b[j,:])

		return kappa*K.T
	def linear_kernel(self,a, b, **kwargs):
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

		a = a[:,group]
		b = b[:,group]
		return kappa*(b @a.T)

	def random_map_kernel(self,a,b, **kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa
		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group



		a = a[:,group]
		b = b[:,group]

		if map is not None:
			return kappa*self.linear_kernel(torch.t(self.map.map(a)),torch.t(self.map.map(b))).detach()
		else:
			return kappa*self.linear_kernel(a,b)





	def laplace_kernel(self,a, b,  **kwargs):
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

		a = a[:,group]
		b = b[:,group]
		X, Y = check_pairwise_arrays(a.numpy(),b.numpy())
		K = - manhattan_distances(a, b) / gamma**2
		K = np.exp(K)  # exponentiate K in-place
		return kappa*torch.from_numpy(K).T



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

		a = a[:,group]
		b = b[:,group]
		#	print (a.shape, b.shape)
		normx = torch.sum(a ** 2, dim=1).view(-1, 1)
		normy = torch.sum(b ** 2, dim=1).view(-1, 1)

		product = torch.mm(b, torch.t(a))
		#sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy
		arg = (-0.5 / (gamma * gamma)) * sqdist
		res = torch.exp(arg)
		return kappa*res

	def covar_kernel(self,a,b, **kwargs):
		"""
		:param a:
		:param b:
		:param cov:
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

		a = a[:,group]
		b = b[:,group]
		a = torch.mm(a, cov)
		b = torch.mm(b, cov)

		normx = torch.sum(a ** 2, dim=1).reshape(-1, 1)
		normy = torch.sum(b ** 2, dim=1).reshape(-1, 1)

		product = torch.mm(b, torch.t(a))
		# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy
		arg = - 0.5 * sqdist
		res = torch.exp(arg)
		return kappa*res


	def ard_kernel(self,a,b, **kwargs):

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

		a = a[:,group]
		b = b[:,group]

		D = torch.diag(1./(gamma[group]))
		a = torch.mm(a,D)
		b = torch.mm(b,D)
		normx = torch.sum(a ** 2, dim=1).reshape(-1, 1)
		normy = torch.sum(b ** 2, dim=1).reshape(-1, 1)

		product = torch.mm(b, torch.t(a))
		# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
		sqdist = -2 * product + torch.t(normx) + normy
		arg = - 0.5 * sqdist
		res = torch.exp(arg)
		return kappa*res


	def ard_kernel_additive(self,a,b, **kwargs):
		if 'groups' in kwargs.keys():
			groups = kwargs['groups']
		else:
			groups = self.groups

		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:,group]
		b = b[:,group]

		(n, z) = tuple(a.size())
		(q, m) = tuple(b.size())

		r = torch.zeros(size=(q, n), dtype  = torch.float64)

		for group_add in groups:
			kwargs['group'] = group_add
			r = r + self.ard_kernel(a, b, **kwargs)

		r = r/float(len(groups))
		return r

	def tanh_kernel(self,a,b, **kwargs):
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

		a = a[:,group]
		b = b[:,group]

		X, Y = check_pairwise_arrays(a.numpy(), b.numpy())
		K = manhattan_distances(a.numpy(), b.numpy())
		K = K.T
		eps = 10e-10
		q = 3
		A = (np.tanh(K) ** q) / (eps + K ** q)
		return kappa*torch.from_numpy(A)

	def angsim_kernel(self,a, b, **kwargs):
		if 'kappa' in kwargs.keys():
			kappa = kwargs['kappa']
		else:
			kappa = self.kappa

		return kappa* (2. / np.pi) * np.arcsin((a.dot(b)) / (a.norm() * b.norm()))

	def polynomial_kernel(self,a,b, **kwargs):
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

		a = a[:,group]
		b = b[:,group]

		K = (torch.mm(b, torch.t(a)) + 1)**power
		return kappa* K

	def polynomial_additive_kernel(self,a,b, **kwargs):

		if 'groups' in kwargs.keys():
			groups = kwargs['groups']
		else:
			groups = self.groups
		if 'group' in kwargs.keys():
			group = kwargs['group']
		else:
			group = self.group

		a = a[:,group]
		b = b[:,group]

		(n, z) = tuple(a.size())
		(q, m) = tuple(b.size())
		no_groups = float(len(groups))
		r = torch.zeros(size=(q, n), dtype  = torch.float64)
		for i,group in enumerate(groups):
			z = self.polynomial_kernel(a[:, group], b[:, group], **kwargs)
			r = r+z
		r = r/no_groups
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

		a = a[:,group].numpy()
		b = b[:,group].numpy()

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
		return torch.from_numpy(K)

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

		a = a[:,group]
		b = b[:,group]

		d = X.size()[1]
		# Z = np.ones(shape = (X.shape[0],Y.shape[0]))
		Z = torch.ones(size=(Y.size()[0], X.size()[0]), dtype = torch.float64)
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
		return Z










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

	def get_1_der(self, point, X):
		"""

		"""
		d = point.size()[1]
		n = X.size()[0]

		print(d, n)
		print(point[0, :])
		res = torch.zeros(size=(n, d), dtype=torch.float64)
		if self.optkernel == "squared_exponential":
			for i in range(d):
				res[:, i] = - 1. / self.gamma ** 2 * (float(point[:, i]) - X[:, i])
			res = res * self.squared_exponential_kernel(point, X, self.gamma, self.kappa)
		else:
			raise AssertionError("Not implemented for this kernel")

		return res

	def get_2_der(self, point):
		"""

		"""
		res = 0
		d = point.size()[1]
		if self.optkernel == "squared_exponential":
			I = torch.eye(d, d, dtype=torch.float64)
			res = 1. / self.gamma ** 2 * I * self.squared_exponential_kernel(point, point, self.gamma, self.kappa)
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
