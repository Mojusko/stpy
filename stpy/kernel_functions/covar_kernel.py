import torch
from stpy.kernel_functions.kernel_params import KernelParams

def covar_kernel(a, b, **kwargs):
	p = KernelParams(kwargs)
	p.assert_existence(["cov", "kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]
	a = torch.mm(a, p.cov)
	b = torch.mm(b, p.cov)

	normx = torch.sum(a ** 2, dim=1).reshape(-1, 1)
	normy = torch.sum(b ** 2, dim=1).reshape(-1, 1)
	product = torch.mm(b, torch.t(a))

	sqdist = -2 * product + torch.t(normx) + normy
	arg = - 0.5 * sqdist
	res = torch.exp(arg)
	return p.kappa * res