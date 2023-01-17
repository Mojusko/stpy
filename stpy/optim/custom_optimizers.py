import cvxpy as cp
import mosek
import numpy as np
import torch


def bisection(g, a, b, N, version='stop'):
	'''Approximate solution of g(x)=0 on interval [a,b] by bisection method.

	Parameters
	----------
	g : function
		The function for which we are trying to approximate a solution g(x)=0.
	a,b : numbers
		The interval in which to search for a solution. The function returns
		None if g(a)*g(b) >= 0 since a solution is not guaranteed.
	N : (positive) integer
		The number of iterations to implement.

	Returns
	-------
	x_N : number
		The midpoint of the Nth interval computed by the bisection method. The
		initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
		midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
		If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
		iteration, the bisection method fails and return None.

	Examples
	--------
	>>> f = lambda x: x**2 - x - 1
	>>> bisection(f,1,2,25)
	1.618033990263939
	>>> f = lambda x: (2*x - 1)*(x - 3)
	>>> bisection(f,0,1,10)
	0.5
	'''
	d = {}

	def f(x):
		if x in d:
			return d[x]
		else:
			d[x] = g(x)
			return d[x]

	if version == 'stop':
		if f(a) < 0.:
			return a
	if f(a) * f(b) > 0.:
		print("Bisection method fails.")
		return None

	a_n = a
	b_n = b
	dict = {}
	for n in range(1, N + 1):
		m_n = (a_n + b_n) / 2.
		f_m_n = f(m_n)
		if f(a_n) * f_m_n < 0:
			a_n = a_n
			b_n = m_n
		elif f(b_n) * f_m_n < 0:
			a_n = m_n
			b_n = b_n
		elif f_m_n == 0:
			print("Found exact solution.")
			return m_n
		else:
			return a_n
			print("Bisection method fails.")
			return None
	return (a_n + b_n) / 2.


def greedy_per_step(fun, add, ground_set, min=True):
	scores = []
	for elem in range(ground_set.size()[0]):
		new = add(ground_set[elem, :].view(1, -1))
		scores.append(fun(new))
	if min:
		j = np.argmin(scores)
	else:
		j = np.argmax(scores)
	return [j]


def QPQC_problem(A, a, s, Sigma=None):
	n = A.shape[0]
	if Sigma is None:
		I = np.eye(n)
		Sigma = I

	# SDP relaxation
	M = np.zeros(shape=(n + 1, n + 1))

	M[0, 1:] = -a.reshape(-1)
	M[1:, 0] = -a.T.reshape(-1)
	M[1:, 1:] = A

	# print (M)

	Meqconst = np.eye(n + 1)
	Meqconst[1:, 1:] = Sigma
	Meqconst[0][0] = 0

	# print (Meqconst)

	First = np.zeros(shape=(n + 1, n + 1))
	First[0, 0] = 1.

	X = cp.Variable((n + 1, n + 1))

	objective = cp.Maximize(cp.trace(M @ X))

	constraints = [X >> 0]
	constraints += [cp.trace(Meqconst @ X) >= s]
	constraints += [cp.trace(First @ X) == 1]

	prob = cp.Problem(objective, constraints)
	prob.solve()

	# print (X.value[1:,1:])
	eigvals, eigvecs = np.linalg.eig(X.value[1:, 1:])

	index = np.argmax(eigvals)
	val = np.max(eigvals)
	x = np.real(eigvecs[index] * np.sqrt(val))
	return val, x


def convex_QCQP(A, a, s, Sigma=None, threads=4, verbose=False):
	"""
	Solving

	min xAx - 2ax
	s.t. xSigmax \leq s
	A, Sigma psd.

	:param A:
	:param a:
	:param s:
	:param Sigma:
	:return:
	"""
	n = A.shape[0]

	if Sigma is None:
		I = np.eye(n)
		Sigma = I

	x = cp.Variable(n)
	objective = cp.Minimize(cp.quad_form(x, A) - 2 * x @ a)
	zero = np.zeros(n)
	# constraints = [ cp.SOC(zero@x + np.array([np.sqrt(s)]), Sigma @ x)]
	constraints = [cp.quad_form(x, Sigma) <= s]
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.MOSEK, mosek_params={mosek.iparam.num_threads: threads,
											  mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
											  mosek.dparam.intpnt_co_tol_pfeas: 1e-8,
											  mosek.dparam.intpnt_co_tol_dfeas: 1e-8,
											  mosek.dparam.intpnt_co_tol_rel_gap: 1e-8},
			   verbose=verbose)

	x_no_const = x.value.reshape(-1, 1)
	val = prob.value
	return val, x_no_const


def QCQP_problem(A, a, s, Sigma=None, threads=4, verbose=False):
	"""
	Solving

	min xAx - 2ax
	s.t. xSigmax == s


	:param A:
	:param a:
	:param s:
	:param Sigma:
	:return:
	"""
	n = A.shape[0]
	lam = cp.Variable(1)
	if Sigma is None:
		I = np.eye(n)
		Sigma = I

	objective = cp.Maximize(lam * s - cp.matrix_frac(a, A - lam * Sigma))
	constraints = [A - lam * Sigma >> 0]
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.MOSEK, mosek_params={mosek.iparam.num_threads: threads,
											  mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
											  mosek.dparam.intpnt_co_tol_pfeas: 1e-12,
											  mosek.dparam.intpnt_co_tol_dfeas: 1e-12,
											  mosek.dparam.intpnt_co_tol_rel_gap: 1e-12},
			   verbose=verbose)

	x_no_const = np.linalg.solve(A - lam.value * Sigma, a)
	val = prob.value
	return val, x_no_const


def solve_mpi(Q, c, tau, verbose=True, up=None, low=None, xwarm=None):
	"""
	Solve MIP program


	"""
	if verbose == True:
		print("Starting Acq. Fucn solver...")
		print("Resolution: ")
	# Grid

	# tau = torch.from_numpy(np.arange(-n, n + 1, 1).astype(np.double)) / n
	N = tau.size()[0]
	d = Q.size()[0]
	s = torch.ones(N)
	Tau = torch.zeros(size=(d, d * N), dtype=torch.float64)
	S = torch.zeros(size=(d, d * N), dtype=torch.float64)

	for j in range(d):
		Tau[j, j * N:(j + 1) * N] = tau
		S[j, j * N:(j + 1) * N] = s

	B = Q @ Tau

	if (up is not None) or (low is not None):
		G = torch.cat((B, -B, S, -S, torch.t(c), -torch.t(c)))
		h = torch.ones(4 * d + 2)
		h[0:2 * d] = 1
		h[3 * d:4 * d] = -1
		h[4 * d] = up
		h[4 * d + 1] = -low
	else:
		G = torch.cat((B, -B, S, -S))
		h = torch.ones(4 * d)
		h[0:2 * d] = 1
		h[3 * d:4 * d] = -1
	# Indicator variables

	x = cp.Variable(d * N, boolean=True)
	if xwarm is not None:
		x.value = xwarm.detach().numpy()
	c = c.view(-1).detach().numpy()

	objective = cp.Minimize(-c * x)
	constraints = [0 <= x, x <= 1, G.detach().numpy() * x <= h.view(-1).detach().numpy()]
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.MOSEK, verbose=verbose, warm_start=True)

	# print (x.value)

	return (torch.from_numpy(Tau.numpy() @ x.value), np.dot(c, x.value))


def newton_solve(f, x0, eps=1e-3, maxiter=100, verbose=False, grad=None):
	"""
	>>> newton_solve(lambda x: x**2,torch.Tensor([2.0,1.0]).double().view(-1))
	tensor([0., 0.], dtype=torch.float64)
	"""
	lam = 1.
	d = int(x0.size()[0])
	x0.requires_grad_(True)
	x = torch.zeros(size=(d, 1), requires_grad=True).view(-1).double()
	x.data = x0.data
	res = f(x) ** 2
	i = 0
	s = 1.

	while torch.max(res) > eps and i < maxiter:
		i = i + 1

		if grad is None:
			nabla_f = torch.autograd.functional.jacobian(f, x, strict=True)
		else:
			nabla_f = grad(x)

		if verbose:
			print(i, "err:", torch.max(res), s)
			print(nabla_f.size())
			print("-----------------------")

		xn = x - torch.linalg.solve(nabla_f + torch.eye(d).double() * s, f(x).view(-1, 1)).view(-1)
		resn = f(xn) ** 2

		if torch.max(resn) < torch.max(res):
			x = xn.requires_grad_(True)
			# lam = np.minimum(lam * 2,1)
			s = s / 2
			res = resn

		else:
			s = s * 2
	# lam = lam /2.
	return x


def matrix_recovery_hermitian_trace_regression(X, b, eps=1e-5):
	"""

	:param X: list of matrices
	:param b: vector of resposnes
	:param eps: constraint tolerance
	:return: reocvered matrix
	"""

	d = X[0].shape[0]
	N = len(X)
	Z = cp.Variable((d, d), symmetric=True)

	constraints = [Z >> 0]
	constraints += [
		cp.trace(X[i] @ Z) >= b[i] - eps for i in range(N)
	]
	constraints += [
		cp.trace(X[i] @ Z) <= b[i] + eps for i in range(N)
	]

	prob = cp.Problem(cp.Minimize(cp.norm(Z, "nuc")),
					  constraints)

	prob.solve()

	return Z.value


if __name__ == "__main__":
	newton_solve(lambda x: x ** 2, torch.Tensor([2.0, 1.0]).double().view(-1), verbose=True)
