import copy

import numpy as np
import torch


def transform(X, low=-1, high=1, functions=True, offsets=None):
	n, d = X.size()
	Y = X.clone()
	transforms = []
	inv_transforms = []

	for i in range(d):

		if offsets is None:
			xmin = torch.min(X[:, i]).clone().numpy()
			xmax = torch.max(X[:, i]).clone().numpy()
		else:
			xmin = offsets[i][0]
			xmax = offsets[i][1]

		k = copy.copy(float((xmin - xmax) / ((low - high))))
		q = copy.copy(float(xmin - low * k))

		k2 = copy.copy(float((low - high) / (xmin - xmax)))
		q2 = copy.copy(float(high - xmax * k2))

		inv_transform = lambda a, k=k, q=q: k * a + q
		transform = lambda a, k2=k2, q2=q2: k2 * a + q2

		transforms.append(copy.copy(transform))
		inv_transforms.append(copy.copy(inv_transform))

		Y[:, i] = torch.from_numpy(np.apply_along_axis(transform, 0, X[:, i].numpy()))

	trans = lambda Z: torch.stack(
		[torch.from_numpy(np.apply_along_axis(transforms[i], 0, Z[:, i].numpy())) for i in range(d)]).T
	inv_trans = lambda Y: torch.stack(
		[torch.from_numpy(np.apply_along_axis(inv_transforms[i], 0, Y[:, i].numpy())) for i in range(d)]).T

	if functions == True:
		return Y, trans, inv_trans, transforms, inv_transforms
	else:
		return Y
