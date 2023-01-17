from stpy.kernel_functions.kernel_params import KernelParams
from stpy.kernel_functions.linear_kernel import linear_kernel

def custom_map_kernel(a, b, **kwargs):
	p = KernelParams(kwargs)
	p.assert_existence(["map", "kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]

	if map is not None:
		return p.kappa * linear_kernel(torch.t(p.map(a)), torch.t(p.map(b))).detach()
	else:
		return p.kappa * linear_kernel(a, b)