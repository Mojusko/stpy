import torch
from pymanopt import Problem
from pymanopt.optimizers import SteepestDescent
from scipy.optimize import minimize


def optimize(manifold, cost_function, number_args, sampling_func, optimizer, restarts):
	[cost_numpy, egrad_numpy, ehess_numpy] = cost_function.define()

	if optimizer == "pymanopt":
		problem = Problem(manifold=manifold, cost=cost_numpy, egrad=egrad_numpy, ehess=ehess_numpy, verbosity=1)
		solver = SteepestDescent(maxiter=100, mingradnorm=1e-8, minstepsize=1e-10)

		def solve(problem, x=None):
			return solver.solve(problem, x=x)

	elif optimizer == "scipy":
		problem = None

		def solve(problem, x=None):
			res = minimize(cost_numpy, xinit, method="L-BFGS-B", jac=egrad_numpy, tol=0.0001)
			return res.x
	else:
		raise NotImplementedError

	# optimization
	repeats = restarts
	best = 10e10
	best_params = [i for i in range(number_args)]

	for _ in range(repeats):
		xinit = sampling_func()
		# try:
		Xopt = solve(problem, x=xinit)
		print(xinit)
		cost = cost_numpy(Xopt)
		print("Run:", _, " cost: ", cost)
		if cost < best:
			best = cost
			if len(best_params) > 1:
				for j in range(number_args):
					best_params[j] = torch.from_numpy(Xopt[j])
			else:
				best_params[0] = torch.from_numpy(Xopt)
	return best_params
