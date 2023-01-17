import numpy as np
import torch
from scipy import integrate

from stpy.helpers.helper import cartesian


def integrate_sin_sin(a, b, omega1, omega2):
	"""

	:param a:
	:param b:
	:param omega1:
	:param omega2:
	:return:
	>>> np.round(integrate_sin_sin(0.2,0.5,2,3),6)
	0.164678
	"""
	eps = 10e-5
	if np.abs(omega1 - omega2) < eps:
		F = lambda x: x / 2 - np.sin(2 * omega1 * x) / (4 * omega1)
	else:
		F = lambda x: (omega2 * np.sin(omega1 * x) * np.cos(x * omega2) -
					   omega1 * np.cos(omega1 * x) * np.sin(omega2 * x)) / (omega1 ** 2 - omega2 ** 2)
	return F(b) - F(a)


def integrate_sin_cos(a, b, omega1, omega2):
	"""

	:param a:
	:param b:
	:param omega1:
	:param omega2:
	:return:
	>>> np.round(integrate_sin_cos(0.2,0.5,2,3),6)
	0.082903
	"""
	eps = 10e-5
	if np.abs(omega1 - omega2) < eps:
		F = lambda x: -np.cos(omega1 * x) ** 2 / (2 * omega1)
	else:
		F = lambda x: -(omega2 * np.sin(omega1 * x) * np.sin(x * omega2) +
						omega1 * np.cos(omega1 * x) * np.cos(omega2 * x)) / (omega1 ** 2 - omega2 ** 2)
	return F(b) - F(a)


def integrate_cos_cos(a, b, omega1, omega2):
	"""

	:param a:
	:param b:
	:param omega1:
	:param omega2:
	:return:
	>>> np.round(integrate_cos_cos(0.2,0.5,2,3),6)
	0.116078
	"""
	eps = 10e-5
	if np.abs(omega1 - omega2) < eps:
		F = lambda x: x / 2 + np.sin(2 * omega1 * x) / (4 * omega1)
	else:
		F = lambda x: (omega1 * np.sin(omega1 * x) * np.cos(x * omega2) -
					   omega2 * np.cos(omega1 * x) * np.sin(omega2 * x)) / (omega1 ** 2 - omega2 ** 2)
	return F(b) - F(a)


def integrate2d_sin_sin(A, B, C, D, a, b, c, d):
	Cos = lambda x: np.cos(x)
	val = (1 / (2 * (b - d) * (b + d))) * (-(((b + d) * (Cos(a * A - A * c + b * C - C * d) -
														 Cos(a * B - B * c + b * C - C * d))) / (a - c)) + (
													   (b + d) * (Cos(a * A - A * c + b * D - d * D) -
																  Cos(a * B - B * c + b * D - d * D))) / (a - c) + (
													   1 / (
													   a + c)) * (b - d) * (Cos(A * (a + c) + C * (b + d)) - Cos(
		B * (a + c) + C * (b + d)) - Cos(A * (a + c) + (b + d) *
										 D) + Cos(B * (a + c) + (b + d) * D)))
	return val


def integrate2d_sin_cos(A, B, C, D, a, b, c, d):
	Sin = lambda x: np.sin(x)
	val = (1 / (2 * (b - d) * (b + d))) * (((b + d) * (-Sin(a * A - A * c + b * C - C * d) +
													   Sin(a * B - B * c + b * C - C * d))) / (a - c) + (
													   (b + d) * (Sin(a * A - A * c + b * D - d * D) -
																  Sin(a * B - B * c + b * D - d * D))) / (a - c) - (
													   1 / (a + c)) * (b - d) * (Sin(A * (a + c) + C * (b + d)) -
																				 Sin(B * (a + c) + C * (b + d)) - Sin(
				A * (a + c) + (b + d) * D) +
																				 Sin(B * (a + c) + (b + d) * D)))
	return val


def integrate2d_cos_cos(A, B, C, D, a, b, c, d):
	Cos = lambda x: np.cos(x)
	val = -(1 / (2 * (b - d) * (b + d))) * (((b + d)(Cos(a * A - A * c + b * C - C * d) -
													 Cos(a * B - B * c + b * C - C * d))) / (
													a - c) - ((b + d) * (Cos(a * A - A * c + b * D - d * D) -
																		 Cos(a * B - B * c + b * D - d * D))) / (
														a - c) + (1 / (
			a + c)) * (b - d) * (Cos(A * (a + c) + C * (b + d)) -
								 Cos(B * (a + c) + C * (b + d)) - Cos(A * (a + c) + (b + d) * D) + Cos(
				B * (a + c) + (b + d) * D)))
	return val


def integrate_sin_multidimensional(a, b, omegas):
	"""

	:param a: bounds start
	:param b: bounds end
	:param omegas: frequencies
	:return:
	>>> np.round(integrate_sin_multidimensional(np.array([0.5]),np.array([1.]),np.array([2.])),5)
	0.47822
	>>> np.round(integrate_sin_multidimensional(np.array([0.5,0.3]),np.array([1.,4.]),np.array([2.,5.])),5)
	-0.01037
	>>> np.round(integrate_sin_multidimensional(np.array([0.5,0.3,0.8]),np.array([1.,4.,3.1]),np.array([2.,5.,1.5])),5)
	0.02232
	"""
	d = omegas.shape[0]

	z = np.array([omegas * b, omegas * a])
	sign = np.array([omegas * 0, omegas * 0 + 1])
	ar = cartesian([z[:, i] for i in range(z.shape[1])])
	signs = cartesian([sign[:, i] for i in range(sign.shape[1])])
	signs = np.sum(signs, axis=1)
	ar = np.sum(ar, axis=1)
	k = 1. / np.prod(omegas)
	# print (ar)

	if d % 2 == 1:
		r = np.cos(ar)
		if d % 4 == 1:
			r = -r
		for i in range(r.shape[0]):
			if signs[i] % 2 == 1:
				r[i] = -r[i]
	else:
		r = np.sin(ar)
		if d % 4 == 3:
			r = -r
		for i in range(r.shape[0]):
			if signs[i] % 2 == 0:
				r[i] = -r[i]
	return k * np.sum(r)


def integrate_cos_multidimensional(a, b, omegas):
	"""

	:param a: bounds start
	:param b: bounds end
	:param omegas: frequencies
	:return:
	>>> np.round(integrate_cos_multidimensional(np.array([0.5]),np.array([1.]),np.array([2.])),5)
	0.03391
	>>> np.round(integrate_cos_multidimensional(np.array([0.5,0.3]),np.array([1.,4.]),np.array([2.,5.])),5)
	0.03169
	>>> np.round(integrate_cos_multidimensional(np.array([0.5,0.3,0.8]),np.array([1.,4.,3.1]),np.array([2.,5.,1.5])),5)
	-0.03784
	"""
	d = omegas.shape[0]

	z = np.array([omegas * b, omegas * a])
	sign = np.array([omegas * 0, omegas * 0 + 1])
	# print(z)
	ar = cartesian([z[:, i] for i in range(z.shape[1])])
	signs = cartesian([sign[:, i] for i in range(sign.shape[1])])
	signs = np.sum(signs, axis=1)
	ar = np.sum(ar, axis=1)
	k = 1. / np.prod(omegas)
	# print (ar)

	if d % 2 == 1:
		r = np.sin(ar)
		if d % 4 == 3:
			r = -r
		for i in range(r.shape[0]):
			if signs[i] % 2 == 1:
				r[i] = -r[i]
	else:
		r = np.cos(ar)
		if d % 4 == 1:
			r = -r
		for i in range(r.shape[0]):
			if signs[i] % 2 == 0:
				r[i] = -r[i]

	return k * np.sum(r)


def romberg2d(func, x1, x2, y1, y2):
	"""

	:param func:
	:param x1:
	:param x2:
	:param y1:
	:param y2:
	:return:
	>>> np.round(romberg2d(lambda x,y:2*x**2+y**2,0,1,1,2),5)
	3.0
	"""
	func2 = lambda y, a, b: integrate.romberg(func, a, b, args=(y,))
	return integrate.romberg(func2, y1, y2, args=(x1, x2))


def quadvec2(func, x1, x2, y1, y2, epsabs=1e-200, epsrel=1e-08, limit=1000, workers=1, quadrature='gk21'):
	"""
	>>> alpha = np.linspace(0.0, 2.0, num=30)
	>>> np.round(quadvec2(lambda x,y: x**alpha + y**alpha,0,1,1,2)[0],5)
	2.0
	>>> np.round(quadvec2(lambda x,y: 2*x**alpha + y**alpha,0,1,1,2)[-1],5)
	3.0
	"""
	func2 = lambda y: \
	integrate.quad_vec(lambda x: func(x, y), x1, x2, epsabs=epsabs, epsrel=epsrel, limit=limit, quadrature=quadrature)[
		0]
	res = integrate.quad_vec(func2, y1, y2, epsabs=epsabs, epsrel=epsrel, limit=limit, quadrature=quadrature)
	return res[0]


def AvgEig(Phi, xtest):
	n = Phi(xtest[0].view(1, -1)).size()[0]
	A = torch.zeros(size=(n, n), dtype=torch.float64)
	for x in xtest:
		v = Phi(x.view(1, -1)).view(-1, 1)
		A = A + v @ v.T
	A = A / xtest.size()[0]
	# import matplotlib.pyplot as plt
	# plt.imshow(A)
	# plt.colorbar()
	# plt.show()
	maxeig = torch.min(torch.symeig(A)[0])
	return maxeig


def volume_eig(Phi, xtest, alpha=0.5):
	n = Phi(xtest[0].view(1, -1)).size()[0]
	A = torch.zeros(size=(n, n), dtype=torch.float64)
	for x in xtest:
		v = Phi(x.view(1, -1)).view(-1, 1)
		mineig = torch.min(torch.symeig(v @ v.T)[0])
		print(mineig)
	vol = 0
	return vol


def chebyschev_nodes(n, d=1, L_infinity_ball=1):
	nodes, w = np.polynomial.chebyshev.chebgauss(n)
	arrays = [nodes.reshape(n, 1) for i in range(d)]
	xtest = cartesian(arrays)
	return xtest


if __name__ == "__main__":
	pass
