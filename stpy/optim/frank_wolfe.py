import torch
import numpy as np
from scipy.optimize import minimize_scalar

def step_frank_wolfe_simplex(F,nablaF, x):
	d = x.shape[0]
	nabla = nablaF(x)
	index = np.argmax(nabla)
	e = np.zeros(d)
	e[index] = 1.
	fn = lambda h: -F( x*h + (1-h)*e)
	res = minimize_scalar(fn,bounds=(10e-8,1-10e-8),method='bounded')
	gamma = res.x
	x =  x*gamma + (1-gamma)*e
	return x

def step_exponential_gradient_descent(nablaF,x,eta = 1.):
	"""

	:param nablaF:
	:param x:
	:param eta:
	:return:
	"""
	x = x*torch.exp(eta*nablaF(x))
	x = x/torch.sum(x)
	return x

def step_wa_simlex(F,nablaF, x, optimality):
	d = x.shape[0]
	nabla = nablaF(x)
	e_plus = np.max(nabla)
	e_minus = np.min(nabla)
	i_minus = np.argmin(nabla)
	i_plus = np.argmax(nabla)
	e = np.zeros(d)

	if (e_plus - optimality)/optimality > (optimality - e_minus)/optimality:
		index = i_plus
		e[index] = 1.
		fn = lambda h: -F( x*h + (1-h)*e)
		res = minimize_scalar(fn,bounds=(10e-8,1-10e-8),method='bounded')
		gamma = res.x
		x =  x*gamma + (1.-gamma)*e
	else:
		index = i_minus
		e[index] = 1.
		fn = lambda h: -F( (x+h*e)/(1+h))
		#res = minimize_scalar(fn,bounds=(0.,1/(1-x[index])),method='bounded')
		res = minimize_scalar(fn,bounds=(-x[index],1-x[index]),method='bounded')
		gamma = res.x
		x =  (x + gamma*e)/(1+gamma)
	return x

