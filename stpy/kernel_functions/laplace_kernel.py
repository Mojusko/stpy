import numpy as np
import torch
from sklearn.metrics.pairwise import check_pairwise_arrays, manhattan_distances
from stpy.kernel_functions.kernel_params import KernelParams

def laplace_kernel(a, b, **kwargs):
	p = KernelParams(kwargs)
	p.assert_existence(["gamma", "kappa", "group"])

	a = a[:, p.group]
	b = b[:, p.group]
	K = - manhattan_distances(a, b) / p.gamma ** 2
	K = np.exp(K)  # exponentiate K in-place
	return p.kappa * torch.from_numpy(K).T