import torch
from stpy.kernel_functions.kernel_params import KernelParams

def gibbs_kernel(a, b, **kwargs):
	p = KernelParams(kwargs)
	p.assert_existence(["gamma_fun", "kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]
	#	print (a.shape, b.shape)
	normx = torch.sum(a ** 2, dim=1).view(-1, 1)
	normy = torch.sum(b ** 2, dim=1).view(-1, 1)

	product = torch.mm(b, torch.t(a))
	# sqdist = torch.tile(normx, b.shape[0]).T + torch.tile(normy, a.shape[0]) - 2 * product
	sqdist = -2 * product + torch.t(normx) + normy

	lengthscales = (p.gamma_fun(a) ** 2 + p.gamma_fun(b).T ** 2)

	print(lengthscales)

	arg = (-0.5 / lengthscales) * sqdist
	res = torch.exp(arg)
	return p.kappa * res