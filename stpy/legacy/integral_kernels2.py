import torch
import numpy as np
from scipy.stats import norm
from stpy.helpers.helper import sample_qmc_halton


class IntegralKernel:

	def __init__(self,dataset, s = 0.1):

		self.x = dataset[0]
		self.y = dataset[1]

		self.s = s
		self.gamma = 1.0
		self.distibution = lambda size: torch.from_numpy(np.random.normal(size=size) * (1. / self.gamma))

		self.n = self.x.size()[0]
		self.d = self.x.size()[1]

		self.basis_func = lambda x,theta : torch.cat((torch.cos(torch.mm(theta,x)),torch.sin(torch.mm(theta,x))),1)
		self.size = 2

		self.set = []
		self.weights = []
		self.params = []
		self.active_basis = None



	def set_distribution(self,distibution):
		self.distibution = distibution

	def set_basis_function(self,fun, size ):
		self.basis_func = fun
		self.size = size

	def sample_basis_function(self):
		param = self.distibution(self.d).view(-1,1)
		return [self.get_basis_function(param),param]

	def sample_basis_function_qmc(self, size = 1):
		inv_cum_dist = lambda x: norm.ppf(x) * (1. / 1.)
		params = torch.from_numpy(sample_qmc_halton(inv_cum_dist, size=(size,self.d)))
		return params

	def sample_basis_vector(self):
		fun = self.sample_basis_function()[0]
		return fun(self.x).view(-1)/np.sqrt(self.n)

	def get_basis_function(self,param):
		return lambda x: self.basis_func(param,x)

	def add_to_basis(self,fun,weight,param):
		self.set.append(fun)
		self.weights.append(weight)
		self.params.append(param)


	def empty(self):
		self.active_basis = None
		self.set = []
		self.weights = []
		self.params = []

	def empty_add_random(self):
		self.empty()
		self.random_increase(1)


	def kernel(self,x,y, noise = True):
		value = torch.zeros(x.size()[0],y.size()[0], dtype = torch.float64)

		for index,elem in enumerate(self.set):
			value += torch.mm(elem(x),torch.t(elem(y))) * self.weights[index]


		if noise == True:
			value = value + self.s * self.s* torch.eye(x.size()[0],y.size()[0], dtype=torch.float64)

		return value


	def random_basis(self, size = 1):
		for _ in range(size):
			f,param= self.sample_basis_function()
			self.add_to_basis(f,1.,param)
		self.uniformize_weights()

	def leverage_socre(self, fun):
		v = fun(self.x) / np.sqrt(self.x.size()[0])
		new_set = self.set

	def basis_map_set(self,x,set):
		value = torch.zeros(len(set), x.size()[0] * self.size, dtype=torch.float64)
		for index, elem in enumerate(set):
			value[index, :] = elem(x).view(-1) / np.sqrt(self.n) #* np.sqrt(weights[index])
		return value

	def outer_kernel(self,x):
		Phi = self.basis_map_set(x,self.set)
		value = torch.mm(Phi,torch.t(Phi))
		return value

	def leverage_score(self, fun):

		return 1.0

	def leverage_score_basis(self, size =1 ):
		count = 0

		while count < size:
			fun, param = self.sample_basis_function()
			leverage_score = self.leverage_score(fun)
			q_bar = size

			q = np.random.binomial(q_bar, float(leverage_score))
			if q > 0:
				w = (q / q_bar) / leverage_score

				self.add_to_basis(fun, w, param)
				count += 1
			else:
				pass

		self.normalize_weights()

	def normalize_weights(self):

		#self.weights = np.ones(len(self.set))/len(self.set)
		sum = np.sum(np.array(self.weights))
		self.weights = np.array(self.weights)/sum
		self.weights = self.weights.tolist()
		#print (self.weights)

	def uniformize_weights(self):
		self.weights = np.ones(len(self.set))/len(self.set)
		self.weights = self.weights.tolist()
		#print (self.weights)

if __name__ == "__main__":
	pass