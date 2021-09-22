import torch
import numpy as np
from scipy.stats import norm
from stpy.helpers.helper import interval
from stpy.helpers.helper import sample_qmc_halton
from stpy.continuous_processes.nystrom_fea import NystromFeatures


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

	def basis_func_dataset(self,param):
		return self.basis_func(param,self.x).view(-1)/np.sqrt(self.n)

	def basis_map_set(self,x,set,weights):
		value = torch.zeros(len(set), x.size()[0] * self.size, dtype=torch.float64)
		# print (value.size(),x.size(),self.set[0](x).view(-1).size())
		for index, elem in enumerate(set):
			# print (np.sqrt(np.array(self.weights[index]).astype(complex)))
			value[index, :] = elem(x).view(-1) / np.sqrt(self.n) #* np.sqrt(weights[index])
		return value

	def empty(self):
		self.active_basis = None
		self.set = []
		self.weights = []
		self.params = []

	def empty_add_random(self):
		self.empty()
		self.random_increase(1)

	def basis_map(self,x):
		return self.basis_map_set(x,self.set,self.weights)

	def kernel(self,x,y, noise = True):
		value = torch.zeros(x.size()[0],y.size()[0], dtype = torch.float64)

		for index,elem in enumerate(self.set):
			value += torch.mm(elem(x),torch.t(elem(y))) * self.weights[index]
		if noise == True:
			value = value + self.s * self.s* torch.eye(x.size()[0],y.size()[0], dtype=torch.float64)

		return value

	def outer_kernel(self,x):
		Phi = self.basis_map(x)
		value = torch.mm(Phi,torch.t(Phi))
		return value

	def expected_phi(self,x, base = 10000):
		Ephi = torch.zeros(x.size()[0]*self.size, dtype = torch.float64)
		for _ in range(base):
			Ephi += self.sample_basis_function()[0](x).view(-1)/np.sqrt(self.n)
		Ephi = Ephi/base
		return Ephi

	def expected_phi_squared(self, x, fun, base=10000):
		prod = 0
		v = fun(x).view(-1)/np.sqrt(self.n)
		for _ in range(base):
			sample = self.sample_basis_function()[0](x).view(-1) / np.sqrt(self.n)
			prod += torch.dot(sample,v)**2
		prod = prod/base
		return prod

	def expected_phi_squared_set(self, x, base = 10000):
		v = self.active_basis

		prod = torch.zeros(x.size()[0],)
		for _ in range(base):
			sample = self.sample_basis_function()[0](x).view(-1) / np.sqrt(self.n)
			prod += torch.mm(sample, v) ** 2
		prod = prod / base
		return prod

	def update_basis(self):
		if self.active_basis is None:
			Phi = self.basis_map(self.x)
			self.active_basis = Phi
			W = torch.mm(Phi, torch.t(Phi)) + self.s * self.s * torch.eye(len(self.set), dtype=torch.float64)
			self.W_inv = torch.inverse(W)
		else:
			v = self.set[-1](self.x).view(1,-1)/np.sqrt(self.x.size()[0])
			self.active_basis = torch.cat((self.active_basis,v),dim = 0)
			W = torch.mm(self.active_basis, torch.t(self.active_basis)) + self.s * self.s * torch.eye(len(self.set), dtype=torch.float64)
			self.W_inv = torch.inverse(W)

	"""
		Scores
	"""


	def leverage_score(self,fun, adding = True, weighted = False, variance = True):



		if adding == True:
			print (fun(self.x).size())
			v = fun(self.x) / np.sqrt(self.x.size()[0])
			new_active_basis = torch.cat((self.active_basis, v), dim=0)
			W = torch.mm(new_active_basis, torch.t(new_active_basis)) + self.s * self.s * torch.eye(len(self.set)+1,  dtype=torch.float64)
			W_inv = torch.inverse(W)
			Phi = new_active_basis
		else:
			W_inv = self.W_inv
			Phi = self.active_basis

		if weighted == True:
			S = torch.diag(torch.sqrt(torch.from_numpy(np.array(self.weights))))
			Phi = torch.mm(S,Phi)
		else:
			pass
		# solve leverage score problem
		A = torch.mm(torch.t(Phi),torch.mm(W_inv,Phi))
		rhs = fun(self.x).view(-1,1)/np.sqrt(self.x.size()[0])
		#print (torch.mm(torch.t(rhs),rhs), torch.mm(torch.t(rhs),torch.mm(A,rhs)))
		if variance == True:
			leverage_score = np.abs(torch.mm(torch.t(rhs),rhs) - torch.mm(torch.t(rhs),torch.mm(A,rhs)))/(self.s**2)
		else:
			leverage_score = np.abs(torch.mm(torch.t(rhs), rhs) - torch.mm(torch.t(rhs), torch.mm(A, rhs)))

		return leverage_score

	def bayes_quad_score(self,fun, base = 1000, Ephi = None):
		"""
			Implements score Phi(set,X)E[Phi(x)]K^{-1}E[Phi(x)]Phi(X,set)

		:param fun: new basis function
		:param base: size of the basis to approximate the expected mapping
		:return:
		"""
		if Ephi is None:
			Ephi = self.expected_phi(self.x, base=base).view(-1,1)
		else:
			pass
		new_set = self.set.copy()
		new_set.append(fun)
		new_Phi = self.basis_map_set(self.x,new_set,np.ones(len(new_set)).tolist())
		W = torch.mm(new_Phi,torch.t(new_Phi)) + self.s * self.s * torch.eye(len(new_set), dtype=torch.float64)
		W_inv = torch.inverse(W)
		v = torch.mm(new_Phi,Ephi)
		score = torch.mm(torch.t(v),torch.mm(W_inv,v))
		return score

	def greedy_score(self, candidates):
		K = self.kernel(self.x,self.x, noise = False)
		scores = torch.zeros(len(candidates), dtype = torch.float64)
		for j in range(len(candidates)):
			fun = candidates[j]
			score = torch.norm( torch.mm(fun,torch.t(fun)) - K )
			#print(torch.norm(torch.mm(fun,torch.t(fun))),torch.norm(K))
			scores[j] = score
		return scores

	def herding_score(self,fun, base = 1000, Ephi = None):
		# if Ephi is None:
		# 	Ephi = self.expected_phi(self.x, base=base).view(-1,1)
		# else:
		# 	pass
		#
		phi = fun(self.x).view(-1) / np.sqrt(self.n)
		Phi = self.active_basis
		n,m = Phi.size()
		v = 0.0
		for j in range(n):
			v = v + torch.dot(Phi[j,:],phi)**2
		v = (1./(n+1))*v
		z = self.expected_phi_squared(self.x, fun, base = base)
		r = z - v
		return r


	def variance_scores(self, set = None):
		if set is None:
			Phi = self.basis_map_set(self.x,self.set,np.ones(len(self.set)).tolist())
			W = torch.mm(Phi, torch.t(Phi)) + self.s * self.s * torch.eye(len(self.set),dtype=torch.float64)
		else:
			Phi = self.basis_map_set(self.x, set, np.ones(len(set)).tolist())
			W = torch.mm(Phi, torch.t(Phi)) + self.s * self.s * torch.eye(len(set), dtype=torch.float64)
		W_inv = torch.inverse(W)
		vars = torch.einsum('ji,ij->j', W,W_inv).view(-1,1)
		return vars
	###############################
	## Increasing the basis size ##
	###############################

	def seq_bayes_quad_increase_heuristic(self, size = 1, candidates = 10, base = 100):
		"""
		Implements sequential bayes quadrature with inexact optimization
		:param size:
		:param base:
		:return:
		"""
		Ephi = self.expected_phi(self.x,base = base).view(-1,1)
		for _ in range(size):
			funs = []
			scores = torch.zeros(candidates, dtype=torch.float64)
			params = []
			for j in range(candidates):
				fun, param = self.sample_basis_function()
				leverage_score = self.bayes_quad_score(fun, Ephi = Ephi)
				funs.append(fun)
				scores[j] = leverage_score
				params.append(param)
			argmax = torch.argmax(scores)
			self.add_to_basis(funs[argmax], 1.0, params[argmax])
		self.quadrature_weights()




	# def herding_exact_increase(self, size = 1):
	# 	"""
	# 	Solves exactly the herding problem with a non-linear solver
	# 	:param size: size of the basis to be increase
	# 	:return: None
	# 	"""
	# 	for _ in range(size):
	# 		#fun = lambda x: self.basis_func(param,x)
	# 		p = lambda omega: np.exp(-np.sum(omega ** 2, axis=1).reshape(-1, 1) / 2 * (self.gamma ** 2)) * np.power(
	# 			(self.gamma / np.sqrt(2 * np.pi)), 1.) * np.power(np.pi / 2, 1.)
	# 		ls = lambda param: -self.leverage_score(self.get_basis_function(torch.from_numpy(param).view(-1,1))).numpy()[0]*p(param.reshape(-1,1))[0]
	# 		# plot ls
	#
	#
	# 		# optimize leverage score
	# 		from scipy.optimize import minimize
	# 		start = self.distibution(self.d).view(-1, 1).numpy()
	# 		res = minimize(ls, start , method="L-BFGS-B", tol=0.0000001, bounds=[[-5,5]])
	# 		solution = torch.from_numpy(res.x).view(-1,1)
	#
	# 		#print (start, solution)
	# 		# params = np.linspace(-10, 10, 1000).reshape(-1, 1)
	# 		# lss = []
	# 		#
	# 		# for param in params:
	# 		# 	#print (param, p(param.reshape(-1,1))[0])
	# 		# 	lss.append(ls(param)*p(param.reshape(-1,1))[0])
	# 		# index = np.argmin(np.array(lss))
	# 		# solution = torch.from_numpy(params[index]).view(-1,1)
	# 		# plt.plot(params, lss)
	# 		# plt.plot(start,ls(start),'ro')
	# 		# plt.plot(solution.numpy(),ls(solution.numpy()),'go')
	# 		#plt.show()
	# 		#print(start, solution)
	# 		self.add_to_basis(self.get_basis_function(solution), 1., solution)

	def herding_increase_heuristic(self, size = 1, candidates = 100,  base = 1000):
		"""

			:param size:
			:param base:
			:return:
			"""
		Ephi = self.expected_phi(self.x, base=base)
		for _ in range(size):
			# print (_)
			self.update_basis()
			funs = []
			scores = torch.zeros(candidates, dtype=torch.float64)
			params = []
			for j in range(candidates):
				fun, param = self.sample_basis_function()
				leverage_score = self.herding_score(fun, Ephi=Ephi)
				# print (j, leverage_score)
				funs.append(fun)
				scores[j] = leverage_score
				params.append(param)
			argmax = torch.argmax(scores)
			self.add_to_basis(funs[argmax], 1., params[argmax])
		self.uniformize_weights()

	def herding_increase_heuristic_group(self, size = 1, candidates = 100,  base = 1000):
		"""

		:param size:
		:param base:
		:return:
		"""
		Ephi = self.expected_phi(self.x,base = base)
		for _ in range(size):
			#print (_)
			self.update_basis()
			funs = []
			params = []
			cand = torch.zeros(candidates,self.n*self.size, dtype = torch.float)
			for j in range(candidates):
				fun, param = self.sample_basis_function()
				funs.append(fun)
				cand[j,:] = fun(self.x).view(-1)/np.sqrt(self.n)
			leverage_scores = self.herding_score_group(cand)

			argmax = torch.argmax(leverage_scores)
			self.add_to_basis(funs[argmax], 1., params[argmax])

		self.uniformize_weights()

	def dpp_increase(self, size = 1, candidates = 1000):
		from dppy.finite_dpps import FiniteDPP
		funs = []
		params = []
		cand = torch.zeros(candidates, self.n * self.size, dtype=torch.float64)

		for j in range(candidates):
			fun, param = self.sample_basis_function()
			funs.append(fun)
			params.append(param)
			cand[j, :] = fun(self.x).view(-1) / np.sqrt(self.n)

		# Random feature vectors
		Phi = torch.t(cand)
		L = Phi.numpy().T.dot(Phi.numpy()) + self.s*self.s*torch.eye(candidates,candidates,dtype = torch.float64).numpy()
		DPP = FiniteDPP('likelihood', **{'L': L})
		DPP.flush_samples()
		DPP.sample_exact_k_dpp(size=size)
		sample_ind = DPP.list_of_samples[0]
		for sample in sample_ind:
			self.add_to_basis(funs[sample], 1., params[sample])
		self.uniformize_weights()



	def leverage_score_sampling(self, size = 1):
		count = 0
		self.update_basis()
		while count < size:

			fun, param = self.sample_basis_function()
			leverage_score = self.leverage_score(fun)
			q_bar = size

			q = np.random.binomial(q_bar, float(leverage_score))
			#print(count, q, leverage_score)
			if q > 0:
				w = (q / q_bar) / leverage_score

				self.add_to_basis(fun, w, param)
				self.update_basis()
				#print("adding", w.float(), param)
				count += 1
			else:
				pass
				#print ("reject", q)
		#print ("sum", np.sum(self.weights))
		#self.uniformize_weights()
		#self.quadrature_weights()
		#self.leverage_weights()
		self.normalize_weights()
	# optimize omp weights

	def hermite_quadrature_basis(self, size = 1):
		self.set = []
		self.weights = []
		self.params = []


		(nodes, weights) = np.polynomial.hermite.hermgauss(int(size))
		nodes = torch.from_numpy(np.sqrt(2) * nodes / self.gamma)
		weights = weights / np.sqrt(np.pi)
		#self.weights = weights.tolist()
		#print (self.weights)
		for index in range(size):
			fun = self.get_basis_function(nodes[index].view(self.d,-1))
			self.add_to_basis(fun,weights[index],nodes[index])


	def greedy_increase(self, size = 1, base =100):
		for _ in range(size):
			# print (_)
			self.update_basis()
			funs = []
			params = []
			cand = torch.zeros(base, self.n , self.size, dtype=torch.float64)
			for j in range(base):
				fun, param = self.sample_basis_function()
				funs.append(fun)
				params.append(param)
				cand[j, :] = fun(self.x) #/ np.sqrt(self.n)

			scores = self.greedy_score(cand)
			argmax = torch.argmin(scores)
			self.add_to_basis(funs[argmax], 1., params[argmax])
			self.normalize_weights()
		#print (self.params)

	def random_increase(self, size = 1):
		for _ in range(size):
			f,param= self.sample_basis_function()
			self.add_to_basis(f,1.,param)
		self.uniformize_weights()

	def qmc_increase(self, size = 1):
		params = self.sample_basis_function_qmc(size = size)
		n = params.size()[0]
		for j in range(n):
			param = params[j,:].view(1,-1)
			#print (params)
			self.add_to_basis(self.get_basis_function(param),1.,param)
		self.uniformize_weights()

	def bach_algortihm(self, size = 1, candidates = 100):
		for _ in range(size):
			set = []
			params = []
			for j in range(candidates):
				f, param = self.sample_basis_function()
				set.append(f)
				params.append(param)
			vars = self.variance_scores(set = set)
			index = np.argmax(-vars)
			self.add_to_basis(set[index],1.,params[index])
			vars = self.variance_scores()
		self.weights = vars.view(-1).tolist()
		self.normalize_weights()



	def pca(self, kernel, size = 1):
		if size > self.n:
			size = self.n
		GP = NystromFeatures(kernel, m=torch.Tensor([size]), s=self.s, approx="svd")
		GP.fit_gp(self.x, self.y)
		return GP.outer_kernel()

	def nystrom(self, kernel, size = 1):
		if size > self.n:
			size = self.n
		GP = NystromFeatures(kernel, m=torch.Tensor([size]), s=self.s, approx="uniform")
		GP.fit_gp(self.x, self.y)
		return GP.outer_kernel()

	###########################
	## weights optimization  ##
	###########################


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

	def bayesian_quadrature_weights(self, base = 1000):
		"""
		Bayesian Quadrature weights
			two possible kernels
		:return:
		"""

		phi = fun(self.x).view(-1) / np.sqrt(self.n)
		Phi = self.active_basis
		n,m = Phi.size()

		Z = self.expected_phi_squared_set(self.x, base = base)

		# assemble kernel
		K = self.outer_kernel(self.x)*self.outer_kernel(self.x)
		# invert kernel
		self.weights = torch.mm(torch.mm(Z,torch.pinverse(K)),Z)
		self.weights = self.weights.tolist()



	def leverage_weights(self):

		Phi = self.basis_map(self.x)
		self.active_basis = Phi
		W = torch.mm(Phi, torch.t(Phi)) + self.s * self.s * torch.eye(len(self.set), dtype=torch.float64)
		self.W_inv = torch.inverse(W)

		new_weights = []
		n = len(self.set)
		for fun in self.set:
			leverage_score = self.leverage_score(fun, adding = False, variance = True, weighted= False)
			#print (leverage_score)
			new_weights.append(leverage_score)
		self.weights = new_weights
		self.normalize_weights()


	def leverage_weights_experimental(self,Kinv):

		Phi = self.basis_map(self.x)
		self.active_basis = Phi
		W = torch.mm(Phi, torch.t(Phi)) + self.s * self.s * torch.eye(len(self.set), dtype=torch.float64)
		W_outer = torch.mm(torch.t(Phi),Phi) + self.s * self.s * torch.eye(self.n*2, dtype=torch.float64)
		W_outer_inv = torch.inverse(W_outer)
		self.W_inv = torch.inverse(W)


		print (torch.norm(W_outer-Kinv))

		#print (Kinv)
		new_weights =[]
		n = len(self.set)
		for fun in self.set:
			#leverage_score = self.leverage_score(fun, adding = False, variance = False, weighted= True)
			v = fun(self.x).view(-1,1)/np.sqrt(self.n)
			#print (torch.trace(torch.mm(torch.t(v),v)))
			mat = torch.mm(torch.t(v),torch.mm(W_outer_inv,v))
			#print (mat)
			leverage_score = torch.trace(mat)
			if leverage_score>0.0:
				#print ("Violation!")
				lv = self.leverage_score(fun, adding=False, variance=True, weighted=False)
				print (float(leverage_score),float(lv))
			#new_weights.append(float(2./(n*leverage_score)))
			new_weights.append(1./(n*leverage_score))
		self.weights = new_weights
		self.normalize_weights()
		#print (self.weights)
		#print (self.params)
		#print(self.weights)
	def omp_optimize(self, size = 1):
		pass



if __name__ == "__main__":
	d = 1
	n = 1024
	N = 100
	L_infinity_ball = 1
	s = 0.001
	xtest = torch.from_numpy(interval(n, d))
	# x = torch.from_numpy(np.random.uniform(-L_infinity_ball, L_infinity_ball, size=(N, d)))
	x = torch.from_numpy(np.linspace(-1, 1, N)).view(N, d)
	f = lambda q: torch.sin(torch.sum(q * 4, dim=1)).view(-1, 1)
	y = f(x)

	IK = IntegralKernel([x, y], s=s)
	IK.random_increase(1000)
	IK.uniformize_weights()
	IK.quadrature_weights()

	fun = IK.sample_basis_function()[0]
	print (IK.bayes_quad_score(fun))
