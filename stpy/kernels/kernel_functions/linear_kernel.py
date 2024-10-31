from stpy.kernel_functions.kernel_params import KernelParams

def linear_kernel(a, b, **kwargs):
	"""
	linear kernl
	:param a:
	:param b:
	:param kwargs:
	:return:
	"""
	p = KernelParams(kwargs)
	p.assert_existence(["kappa", "group"])
	a = a[:, group]
	b = b[:, group]
	return kappa * (b @ a.T)