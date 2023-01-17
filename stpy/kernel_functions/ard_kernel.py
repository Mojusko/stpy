import torch
from stpy.kernel_functions.kernel_params import KernelParams


def ard_kernel(a, b, **kwargs):
	p = KernelParams(kwargs)
	p.assert_existence(["ard_gamma", "kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]

	D = torch.diag(1. / (p.ard_gamma[p.group]))

	a = torch.mm(a, D)
	b = torch.mm(b, D)

	normx = torch.sum(a ** 2, dim=1).reshape(-1, 1)
	normy = torch.sum(b ** 2, dim=1).reshape(-1, 1)

	product = torch.mm(b, torch.t(a))
	sqdist = -2 * product + torch.t(normx) + normy
	arg = - 0.5 * sqdist
	res = torch.exp(arg)
	return p.kappa * res


def ard_kernel_diag(a, b, **kwargs):
	p = KernelParams(kwargs)
	p.assert_existence(["ard_gamma", "kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]

	D = torch.diag(1. / (p.ard_gamma[p.group]))
	a = torch.mm(a, D)
	b = torch.mm(b, D)
	normx = torch.sum(a ** 2, dim=1).reshape(-1, 1)
	normy = torch.sum(b ** 2, dim=1).reshape(-1, 1)

	product = torch.mm(b, torch.t(a))
	sqdist = -2 * product + torch.t(normx) + normy
	arg = - 0.5 * sqdist
	res = torch.exp(arg)
	return p.kappa * res


def ard_per_group_kernel_additive(self, a, b, **kwargs):
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
		# kwargs['ard_gamma'] = ard_per_group[groups_index:groups_index+size_group]
		gamma = ard_per_group[groups_index:groups_index + size_group]
		groups_index += size_group

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
	return kappa * r