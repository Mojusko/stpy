import numpy as np
import torch
from stpy.kernel_functions.kernel_params import KernelParams

def squared_exponential_kernel(a, b, **kwargs):
	"""

	:param a:
	:param b:
	:param kwargs: must include gamma, kappa, group
	:return:
	"""
	p = KernelParams(kwargs)
	p.assert_existence(["gamma", "kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]
	#	print (a.shape, b.shape)
	normx = torch.sum(a ** 2, dim=1).view(-1, 1)
	normy = torch.sum(b ** 2, dim=1).view(-1, 1)

	product = torch.mm(b, torch.t(a))
	# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
	sqdist = -2 * product + torch.t(normx) + normy
	arg = (-0.5 / (p.gamma * p.gamma)) * sqdist
	res = torch.exp(arg)
	return p.kappa * res

def squared_exponential_kernel_diag(a,b, **kwargs):
	p = KernelParams(kwargs)
	p.assert_existence(["gamma", "kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]
	sqdist = (a-b)**2
	arg = (-0.5 / (p.gamma * p.gamma)) * sqdist
	res = torch.exp(arg)
	return p.kappa * res