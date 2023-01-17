import cvxpy as cp
import matplotlib.pyplot as plt
import mosek
import numpy as np

from stpy.optim.custom_optimizers import QCQP_problem


def maximum_volume_ellipsoid_l1_polytope_ellipse(ellipse, l1_polytope, verbose=False):
	"""
	ellipse is
	xA_ix + 2b_i x + c_i \leq 0

	\sum q_i | x^\top a_i - b_i |

	:param ellipse:
	:param polytope:
	:param verbose:
	:return:
	"""

	p = ellipse[0].shape[0]

	B = cp.Variable((p, p), PSD=True)
	d = cp.Variable((p, 1))
	lam = cp.Variable((1, 1))
	obj_max = cp.Maximize(cp.log_det(B))

	constraints = []
	A, b, c = ellipse

	eye = np.eye(p)
	zeros = np.zeros(shape=(1, p))
	invA = np.linalg.inv(A)

	constraints.append(
		cp.bmat([
			[-lam - c + b.T @ invA @ b, zeros, d.T + b.T @ invA.T],
			[zeros.T, lam * eye, B],
			[d + invA @ b, B, invA]]) >> 0)

	q, X, y, eps = l1_polytope
	m = X.shape[0]
	t = cp.Variable((m, 1))
	constraints.append(q.T @ t <= eps)
	constraints.append(t >= 0.)
	for i in range(m):
		ai = X[i, :]
		bi = y[i]
		constraints.append(cp.norm2(B @ ai) + ai.T @ d - bi <= t[i])
		constraints.append(cp.norm2(B @ ai) - ai.T @ d + bi <= t[i])

	prob = cp.Problem(obj_max, constraints)
	prob.solve(solver=cp.MOSEK, verbose=verbose)

	print(prob.status)
	if B.value is not None:
		return np.linalg.inv(B.value).T @ np.linalg.inv(B.value), d.value
	else:
		return None, None


def maximum_volume_ellipsoid_relu_polytope_ellipse(ellipse, relu_polytope, verbose=False):
	"""
	ellipse is
	xA_ix + 2b_i x + c_i \leq 0


	(eta_i + x^x_i) \leq eps_i

	:param ellipse:
	:param polytope:
	:param verbose:
	:return:
	"""

	p = ellipse[0].shape[0]

	B = cp.Variable((p, p), PSD=True)
	d = cp.Variable((p, 1))
	lam = cp.Variable((1, 1))
	obj_max = cp.Maximize(cp.log_det(B))

	constraints = []
	A, b, c = ellipse

	eye = np.eye(p)
	zeros = np.zeros(shape=(1, p))
	invA = np.linalg.inv(A)

	constraints.append(
		cp.bmat([
			[-lam - c + b.T @ invA @ b, zeros, d.T + b.T @ invA.T],
			[zeros.T, lam * eye, B],
			[d + invA @ b, B, invA]]) >> 0)

	q, X, y, eps = relu_polytope
	m = X.shape[0]
	t = cp.Variable((m, 1))
	constraints.append(q.T @ t <= eps)
	constraints.append(t >= 0.)
	for i in range(m):
		ai = X[i, :]
		bi = y[i]
		constraints.append(cp.pos(cp.norm2(B @ ai) + ai.T @ d - bi) <= t[i])

	prob = cp.Problem(obj_max, constraints)
	prob.solve(solver=cp.MOSEK, verbose=verbose)

	print(prob.status)
	if B.value is not None:
		return np.linalg.inv(B.value).T @ np.linalg.inv(B.value), d.value
	else:
		return None, None


def maximum_volume_ellipsoid_intersection_ellipsoids(ellipses, planes=None, verbose=False):
	"""
	Each ellipse is
	xA_ix + 2b_i x + c_i \leq 0

	:param elipses: list of [A,b,c]

	:return:elipse  ||x-v||_B^2 < 1
	"""

	p = ellipses[0][0].shape[0]
	m = len(ellipses)

	B = cp.Variable((p, p), PSD=True)
	d = cp.Variable((p, 1))
	lam = cp.Variable((m, 1))

	obj_max = cp.Maximize(cp.log_det(B))

	constraints = []
	for index, ellipse in enumerate(ellipses):
		A, b, c = ellipse

		eye = np.eye(p)
		zeros = np.zeros(shape=(1, p))
		invA = np.linalg.inv(A)

		constraints.append(
			cp.bmat([
				[-lam[index, 0] - c + b.T @ invA @ b, zeros, d.T + b.T @ invA.T],
				[zeros.T, lam[index, 0] * eye, B],
				[d + invA @ b, B, invA]]) >> 0)

	if planes is not None:
		for index, plane in enumerate(planes):
			a, b = plane
			constraints.append(cp.norm2(B @ a) + a.T @ d <= b)

	prob = cp.Problem(obj_max, constraints)
	prob.solve(solver=cp.MOSEK, verbose=verbose)

	print(prob.status)
	if B.value is not None:
		return np.linalg.inv(B.value).T @ np.linalg.inv(B.value), d.value
	else:
		return None, None


# return B.value, -d.value

def ellipsoid_cut(c, B, a, beta):
	"""
	:param c: elipsoid center
	:param B: elipsoid covariance
	:param a: a
	:param beta:

	(x-c)^\top B^{-1} (x-c) \leq 1
	a^x \leq \beta

	:return:
	"""
	N = a.T @ B @ a
	print(N)
	alpha = (a.T @ c - beta) / np.sqrt(N)
	if alpha > 0:
		d = c.shape[0]
		tau = (1 + d * alpha) / (d + 1)
		delta = ((d ** 2) / (d ** 2 - 1)) * (1 - alpha ** 2)
		sigma = (2. * (1 + d * alpha)) / ((d + 1) * (1 + alpha))

		s = B @ a
		c = c + tau * (s / np.sqrt(N))
		B = delta * (B - sigma * (s @ s.T) / (N))
	return (c, B)


def maximize_on_elliptical_slice(x, Sigma, mu, c, l, Lambda, u):
	"""
	solves the problem
		min x^\top \theta
		s.t. (\theta - \mu)Sigma(\theta - \mu) \leq c
		l \leq Lambda \theta \leq u
	"""

	m = x.shape[0]
	zero = np.zeros(m)
	theta = cp.Variable(m)
	obj_max = cp.Maximize(x @ theta)
	Sigma_sqrt = np.linalg.cholesky(Sigma)
	constraints = [cp.SOC(zero.T @ theta + c, Sigma_sqrt @ (theta - mu))]
	constraints.append(Lambda @ theta >= l)
	constraints.append(Lambda @ theta <= u)
	prob = cp.Problem(obj_max, constraints)
	prob.solve(solver=cp.MOSEK, verbose=False
			   , mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
	val = prob.value
	theta = theta.value
	return val, theta


def maximize_matrix_quadratic_on_ellipse(X, Sigma, mu, c, threads=4):
	"""
	solves the problem
		max \theta ^top Z \theta
		s.t. (\theta - \mu)Sigma(\theta - \mu) \leq c
	"""
	a = -X @ mu.reshape(-1)
	val, theta = QCQP_problem(-X, a, c, Sigma=Sigma, threads=threads)
	val = -val + mu @ X @ mu
	return val, theta


def minimize_matrix_quadratic_on_ellipse(Z, Sigma, mu, c, threads=4):
	"""
	solves the problem
		min \theta ^top Z \theta
		s.t. (\theta - \mu)Sigma(\theta - \mu) \leq c
	"""

	m = Z.shape[0]
	zero = np.zeros(m)
	Sigma_sqrt = np.linalg.cholesky(Sigma)
	theta = cp.Variable(m)
	obj = cp.Minimize(cp.quad_form(theta, Z))
	constraints = [cp.SOC(zero.T @ theta + c, Sigma_sqrt @ (theta - mu))]
	prob = cp.Problem(obj, constraints)
	prob.solve(solver=cp.MOSEK, verbose=False,
			   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
							 mosek.iparam.num_threads: threads})
	val = prob.value
	theta = theta.value
	return val, theta


def maximize_quadratic_on_ellipse(x, Sigma, mu, c, threads=4):
	"""
	solves the problem
		max (x^\top \theta)^2
		s.t. (\theta - \mu)Sigma(\theta - \mu) \leq c
	"""
	X = x.reshape(-1, 1) @ x.reshape(1, -1)
	a = -X @ mu.reshape(-1)
	val, theta = QCQP_problem(-X, a, c, Sigma=Sigma, threads=threads)
	val = -val + mu @ X @ mu
	return val, theta


def minimize_quadratic_on_ellipse(x, Sigma, mu, c, threads=4):
	"""
	solves the problem
		min (x^\top \theta)^2
		s.t. (\theta - \mu)Sigma(\theta - \mu) \leq c
	"""

	m = x.shape[0]
	zero = np.zeros(m)
	Sigma_sqrt = np.linalg.cholesky(Sigma)
	theta = cp.Variable(m)
	obj = cp.Minimize((x @ theta) ** 2)
	constraints = [cp.SOC(zero.T @ theta + c, Sigma_sqrt @ (theta - mu))]
	prob = cp.Problem(obj, constraints)
	prob.solve(solver=cp.MOSEK, verbose=False,
			   mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
							 mosek.iparam.num_threads: threads})
	val = prob.value
	theta = theta.value
	return val, theta


def KY_initialization(X):
	(n, d) = X.shape
	y = np.zeros(shape=(d, d,))
	zs = []
	c = np.random.randn(d)
	for j in range(d):
		id_max = np.argmax(X @ c)
		id_min = np.argmin(X @ c)

		z_max = X[np.argmax(X @ c), :]
		z_min = X[np.argmin(X @ c), :]

		zs = zs + [id_max, id_min]
		y[j, :] = z_max - z_min

		c = np.random.randn(d)
		for i in range(j):
			c = c - ((np.dot(c, y[i, :])) / (np.dot(y[i, :], y[i, :]))) * y[i, :]

	mu = np.zeros(shape=(n))
	mu[zs] = 1.
	mu = mu / np.sum(mu)
	return mu


def KY_initialization_modified(X):
	(n, d) = X.shape
	y = np.zeros(shape=(d, d,))
	zs = []
	c = np.random.randn(d)
	for j in range(d):
		id_max = np.argmax(X @ c)
		id_min = np.argmin(X @ c)

		z_max = X[np.argmax(X @ c), :]
		z_min = X[np.argmin(X @ c), :]

		zs = zs + [id_max]
		y[j, :] = z_max - z_min

		c = np.random.randn(d)
		for i in range(j):
			c = c - ((np.dot(c, y[i, :])) / (np.dot(y[i, :], y[i, :]))) * y[i, :]

	mu = np.zeros(shape=(n))
	mu[zs] = 1.
	mu = mu / np.sum(mu)
	return mu


def plot_ellipse(offset, cov, scale=1, theta_num=1000, axis=None, plot_kwargs=None, fill=False, fill_kwargs=None,
				 color='r'):
	'''
	offset = 2d array which gives center of ellipse
	cov = covariance of ellipse
	scale = scale ellipse by constant factor
	theta_num = used for a linspace below, not sure exactly (?)

	'''
	# Get Ellipse Properties from cov matrix

	eig_vec, eig_val, u = np.linalg.svd(cov)
	# Make sure 0th eigenvector has positive x-coordinate
	if eig_vec[0][0] < 0:
		eig_vec[0] *= -1
	semimaj = np.sqrt(eig_val[0])
	semimin = np.sqrt(eig_val[1])
	semimaj *= scale
	semimin *= scale
	phi = np.arccos(np.dot(eig_vec[0], np.array([1, 0])))
	if eig_vec[0][1] < 0 and phi > 0:
		phi *= -1

	# Generate data for ellipse structure
	theta = np.linspace(0, 2 * np.pi, theta_num)
	r = 1 / np.sqrt((np.cos(theta)) ** 2 + (np.sin(theta)) ** 2)
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	data = np.array([x, y])
	S = np.array([[semimaj, 0], [0, semimin]])
	R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
	T = np.dot(R, S)
	data = np.dot(T, data)
	data[0] += offset[0]
	data[1] += offset[1]

	# Plot!
	return_fig = False
	if axis is None:
		axis = plt.gca()

	if plot_kwargs is None:
		p, = axis.plot(data[0], data[1], color=color, linestyle='-')
	else:
		p, = axis.plot(data[0], data[1], **plot_kwargs)

	if fill == True:
		if fill_kwargs is None:
			fill_kwargs = dict()
		axis.fill(data[0], data[1], alpha=0.2, color=color)


if __name__ == "__main__":
	d = 2

	s1 = 1
	s2 = 1

	A1 = np.random.randn(d, d)
	A1 = A1.T @ A1

	A2 = np.random.randn(d, d)
	A2 = A2.T @ A2

	center1 = np.zeros((d, 1))
	center2 = np.ones((d, 1))

	b1 = - A1 @ center1
	b2 = - A2 @ center2

	c1 = -s1 + center1.T @ A1 @ center1
	c2 = -s2 + center2.T @ A2 @ center2

	# ellipsoids = [[A1,b1,c1],[A2,b2,c2]]
	ellipsoids = [[A2, b2, c2]]
	planes = [[center2, np.array([[0.]])]]

	A, b = maximum_volume_ellipsoid_intersection_ellipsoids(ellipsoids, planes=planes)
	# c = 1

	axis = plt.gca()

	## the cov is
	# (x-center)cov^{-1}(x-center)
	# plot_ellipse(np.array([0.,0.]), cov=np.array([[2,0.],[0.0,2.]]), scale = 1., axis=axis, fill=True, color = 'purple')

	plot_ellipse(center1.reshape(-1), cov=np.linalg.inv(A1), scale=1., axis=axis, fill=True)
	plot_ellipse(center2.reshape(-1), cov=np.linalg.inv(A2), scale=1., axis=axis, fill=True, color='b')

	plot_ellipse(b.reshape(-1), cov=np.linalg.inv(A), scale=1., axis=axis, fill=True, color='g')

	plt.xlim([-4, 4])
	plt.ylim([-4, 4])
	plt.show()
