import torch
from scipy.optimize import minimize

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.helpers.helper import interval
from stpy.kernels import KernelFunction
from pymanopt.manifolds import Euclidean

if __name__ == "__main__":
	d = 2
	n = 3


	## Squared exponential with single parameter
	GP = GaussianProcess(gamma=1., kernel_name="ard", d=2)
	x = torch.rand(n,d).double()*2 - 1
	y = GP.sample(x)
	GP.fit_gp(x,y)
	xtest = torch.from_numpy(interval(50,2,L_infinity_ball=1))

	#
	# init_val = None
	# manifold = Euclidean(2)
	# bounds = None
	#
	# params = {"0":{"kappa":(1.,Euclidean(1),None),"ard_gamma":(init_val, manifold, bounds)}}
	#GP.optimize_params_general(params = params, maxiter = 100)

	#GP.optimize_params(type = "bandwidth", restarts=2)


#
	## Additive quick
	k = KernelFunction(kernel_name = "ard", d = 2, groups = [[0],[1]] )
	GP = GaussianProcess(kernel=k)
	x = torch.rand(n,d).double()*2 - 1
	y = GP.sample(x)
	GP.fit_gp(x,y)

	#GP.optimize_params(type="bandwidth", restarts=2)




	# ## Additive via algebra
	k1 = KernelFunction(kernel_name="ard" ,ard_gamma = 0.1, d = 1, group=[0])
	k2 = KernelFunction(kernel_name="polynomial" ,ard_gamma = 0.5, power = 2, d = 1, group=[1])
	k = k1 + k2
	#
	# print (k.params_dict)
	GP = GaussianProcess(kernel=k, d=2)
	#
	x = torch.rand(n, d).double() * 2 - 1
	y = GP.sample(x)
	GP.fit_gp(x, y)
	#GP.optimize_params(type="bandwidth", restarts=2)


	## Additive two the same
	k1 = KernelFunction(kernel_name="ard" ,ard_gamma = 0.1, d = 1, group=[0])
	k2 = KernelFunction(kernel_name="ard" ,ard_gamma = 0.5, power = 2, d = 1, group=[1])
	GP = GaussianProcess(kernel=k, d=2)
	#
	x = torch.rand(n, d).double() * 2 - 1
	y = GP.sample(x)
	GP.fit_gp(x, y)
	#GP.optimize_params(type="bandwidth", restarts=2)


	## Optimize groups
	k = KernelFunction(kernel_name="ard", d=2, groups = [[0,1]])
	GP = GaussianProcess(kernel=k, d=2)
	#
	x = torch.rand(n, d).double() * 2 - 1
	y = GP.sample(x)
	GP.fit_gp(x, y)
	#print(k.params_dict)
	#GP.optimize_params(type="groups", restarts=2)

	## Optimize power in polynomial kernel
	k = KernelFunction(kernel_name="polynomial", d=2, power = 3)
	GP = GaussianProcess(kernel=k, d=2)
	#
	x = torch.rand(n, d).double() * 2 - 1
	y = GP.sample(x)
	GP.fit_gp(x, y)
	#print(k.params_dict)
	params = {"0":{"power":(1.,[1,2,3,4,5],None)}}
	#GP.optimize_params_general(params = params, optimizer="discrete")


	## Covar
	k = KernelFunction(kernel_name="full_covariance_se", d=2)
	GP = GaussianProcess(kernel=k, d=2)
	#
	x = torch.rand(n, d).double() * 2 - 1
	y = GP.sample(x)
	GP.fit_gp(x, y)
	#GP.optimize_params(type="covariance", restarts=2)

	## cova with regularizer
	k = KernelFunction(kernel_name="full_covariance_se", d=2)
	GP = GaussianProcess(kernel=k, d=2)
	#
	x = torch.rand(n, d).double() * 2 - 1
	y = GP.sample(x)
	GP.fit_gp(x, y)
	GP.optimize_params(type="covariance", restarts=2, regularizer=["spectral_norm",0.1])