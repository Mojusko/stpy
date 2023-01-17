from mpmath import *

phi = lambda x: (0 <= x < 1)  # scaling fct
psi = lambda x: (0 <= x < .5) - (.5 <= x < 1)  # wavelet fct
phi_j_k = lambda x, j, k: 2 ** (j / 2) * phi(2 ** j * x - k)
psi_j_k = lambda x, j, k: 2 ** (j / 2) * psi(2 ** j * x - k)


def haar(f, interval, level):
	c0 = quadgl(lambda t: f(t) * phi_j_k(t, 0, 0), interval)

	coef = []
	for j in xrange(0, level):
		for k in xrange(0, 2 ** j):
			djk = quadgl(lambda t: f(t) * psi_j_k(t, j, k), interval)
			coef.append((j, k, djk))

	return c0, coef


def haarval(haar_coef, x):
	c0, coef = haar_coef
	s = c0 * phi_j_k(x, 0, 0)
	for j, k, djk in coef:
		s += djk * psi_j_k(x, j, k)
	return s
