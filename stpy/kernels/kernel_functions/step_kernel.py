from stpy.kernel_functions.kernel_params import KernelParams
import torch

def step_kernel(a, b, **kwargs):
	p = KernelParams(kwargs)
	p.assert_existence(["kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]

	n, d = a.size()
	m, d = b.size()

	K = torch.zeros(size=(n, m)).double()

	for i in range(n):
		for j in range(m):
			K[i, j] = a[i, :] + b[j, :] - torch.abs(a[i, :] - b[j, :])

	return p.kappa * K.T