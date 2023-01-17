import pickle
from abc import ABC
import numpy as np
import torch
import pymanopt
from autograd_minimize import minimize
from pymanopt.manifolds import Product
from pymanopt.optimizers import SteepestDescent
from torchmin import minimize as minimize_torch

from stpy.helpers import helper
from stpy.optim.custom_optimizers import bisection

class Estimator(ABC):

	def log_marginal(self, kernel, X, weight):
		func = kernel.get_kernel()
		K = func(self.x, self.x, **X) + torch.eye(self.n, dtype=torch.float64) * self.s * self.s
		L = torch.linalg.cholesky(K)
		logdet = -0.5 * 2 * torch.sum(torch.log(torch.diag(L))) * weight
		alpha = torch.cholesky_solve(self.y, L)
		logprob = -0.5 * torch.mm(torch.t(self.y), alpha) + logdet
		logprob = -logprob
		return logprob

	def optimize_params_general(self, params={}, restarts=2,
								optimizer="pymanopt", maxiter=1000,
								mingradnorm=1e-4, regularizer_func=None,
								verbose=False, scale=1., weight=1., save = False,
								save_name = 'model.np', parallel = False, cores = None):
		"""

		:param params:
		:param restarts:
		:param optimizer:
		:param maxiter:
		:param mingradnorm:
		:param regularizer_func:
		:param verbose:
		:return:
		"""
		manifolds = []
		bounds = []
		init_values = []

		for key, dict_params in params.items():
			for var_name, value in dict_params.items():
				init_value, manifold, bound = value
				manifolds.append(manifold)
				bounds.append(bound)
				init_values.append(init_value)

		if optimizer == "pymanopt":

			egrad = ehess = None

			@helper.conditional_decorator(pymanopt.autodiff.backends.PyTorch, optimizer == "pymanopt")
			def cost(*args):
				# print (args)
				input_dict = {}
				i = 0
				for key, dict_params in params.items():
					small_param = {}
					for var_name, value in dict_params.items():
						small_param[var_name] = args[i]
						i = i + 1
					input_dict[key] = small_param

				if regularizer_func is not None:
					f = self.log_marginal(self.kernel_object, input_dict, weight) + regularizer_func(args)
				else:
					f = self.log_marginal(self.kernel_object, input_dict, weight)
				return f

			manifold = Product(tuple(manifolds))
			problem = pymanopt.Problem(manifold, cost=cost, egrad=egrad, ehess=ehess, verbosity=int(verbose) + 1)
			solver = SteepestDescent(logverbosity=1, maxiter=maxiter, mingradnorm=mingradnorm)

			# get initial point
			objective_values = []
			objective_params = []

			for rep in range(restarts):
				x_init = []
				for index, man in enumerate(manifolds):
					if init_values[index] is None:
						x_sub = man.rand() * scale
					else:
						x_sub = np.array([init_values[index]])
					x_init.append(x_sub)
				# try:
				x_opt, log = solver.solve(problem, x=x_init)

				objective_params.append(x_opt)
				objective_values.append(log['final_values']['f(x)'])
			# except Exception as e:
			#	print (e)
			#	print ("Optimization restart failed:", x_init)
			# pick the smallest objective
			best_index = np.argmin(objective_values)
			x_opt = [torch.from_numpy(j) for j in objective_params[best_index]]

		elif optimizer == "scipy":
			cost_numpy = lambda x: cost(x).detach.numpy()
			egrad_numpy = lambda x: egrad(x).detach().numpy()

		elif optimizer == "bisection":

			def cost(x):
				input_dict = self.kernel_object.params_dict
				counter = 0
				for key, dict_params in params.items():
					for var_name, value in dict_params.items():
						input_dict[key][var_name] = x
						counter += 1

				if regularizer_func is not None:
					f = self.log_marginal(self.kernel_object, input_dict, weight) + regularizer_func(x)
				else:
					f = self.log_marginal(self.kernel_object, input_dict, weight)
				return f

			a,b = bounds[0]
			x_opt = [bisection(cost,a,b,100)]

		elif optimizer == "pytorch-minimize":
			var_names = []
			dims = [0,]
			for key, dict_params in params.items():
				for var_name, value in dict_params.items():
					init_value, manifold, bound = value

					manifolds.append(manifold)
					bounds.append(bound)
					init_values.append(init_value)
					var_names.append(var_name)
					dims.append(manifold.dim)

			dims = np.cumsum(dims).astype(int)

			def cost(x):
				input_dict = self.kernel_object.params_dict
				counter = 0
				for key, dict_params in params.items():
					for var_name, value in dict_params.items():
						if key != "likelihood":
							input_dict[key][var_name] = x[dims[counter]:dims[counter+1]]
						else:
							self.s = x[dims[counter]:dims[counter+1]]
							counter += 1

				if regularizer_func is not None:
					f = self.log_marginal(self.kernel_object, input_dict, weight) + regularizer_func(x)
				else:
					f = self.log_marginal(self.kernel_object, input_dict, weight)
				return f

			objective_values = []
			objective_params = []
			x_opt = []

			dim = dims[-1]
			self.prepared_log_marginal = False
			for rep in range(restarts):
				#try:
				if init_values[0] is None:
					x_init = torch.randn(size=(dim, 1)).double().view(-1)**2 * scale
				else:
					x_init = init_values[0](dim)

				if bounds[0] is None:
					res = minimize_torch(cost, x_init, method='l-bfgs', tol=1e-10, disp=verbose + 1,
										 options={'max_iter': maxiter, 'gtol':mingradnorm})
					objective_params.append(res.x)
					objective_values.append(res.fun)
				else:
					print ("Constrained optimization with bounds", bounds[0])
					res = minimize(cost, x_init.numpy(), backend='torch', method='L-BFGS-B',
								   bounds=bounds[0], precision='float64', tol=1e-8,
								   options={'ftol': 1e-10,
											'gtol': mingradnorm, 'eps': 1e-08,
											'maxfun': 15000, 'maxiter': maxiter,
											'maxls': 20, 'disp' : verbose + 1})

					objective_params.append(torch.from_numpy(res.x))
					objective_values.append(torch.from_numpy(res.fun))
				#except Exception as e:
				#	print(e)
			# save models

			if save:
				vals = {'params': objective_params,
						'evidence':objective_values,
						'repeats':restarts,
						'dim':dims,
						'param_names':params}

				with open(save_name, 'wb') as f:
					pickle.dump(vals, f)


			best_index = np.argmin(objective_values)

			counter = 0
			for key, dict_params in params.items():
				for var_name, value in dict_params.items():
					x_opt.append(objective_params[best_index][dims[counter]:dims[counter+1]])
					counter += 1

		elif optimizer == "discrete":
			values = []
			configurations = manifolds[0]
			for config in manifolds[0]:
				values.append(cost(config))

			best_index = np.argmin(values)
			x_opt = [configurations[best_index]]
		else:
			raise AssertionError("Optimizer not implemented.")

		# put back into default dic
		i = 0
		for key, dict_params in params.items():
			for var_name, value in dict_params.items():
				if key == "likelihood":
					self.s = x_opt[i]

				else:
					self.kernel_object.params_dict[key][var_name] = x_opt[i]
				i = i + 1

		# print ("--------- Finished. ------------")
		# print (self.kernel_object.params_dict)

		# disable back_prop
		self.back_prop = False

		# refit the model
		self.fit = False
		print(self.description())
		self.fit_gp(self.x, self.y)
		return True

	def load_params(self, objective_params, params, dims):
		self.fig = False
		self.back_prop = False
		x_opt = []
		counter = 0
		for key, dict_params in params.items():
			for var_name, value in dict_params.items():
				x_opt.append(objective_params[dims[counter]:dims[counter + 1]])
				counter += 1

		counter = 0
		for key, dict_params in params.items():
			for var_name, value in dict_params.items():
				self.kernel_object.params_dict[key][var_name] = x_opt[counter]
				counter += 1

		print(self.description())



	def visualize_function(self, xtest, f_trues, filename=None, colors=None, figsize = (15, 7)):
		d = xtest.size()[1]
		if d == 1:
			if isinstance(f_trues, list):
				for f_true in f_trues:
					plt.plot(xtest, f_true(xtest))
			else:
				plt.plot(xtest, f_trues(xtest))
		elif d == 2:
			from scipy.interpolate import griddata
			plt.figure(figsize=figsize)
			plt.clf()
			ax = plt.axes(projection='3d')
			xx = xtest[:, 0].numpy()
			yy = xtest[:, 1].numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]


			if isinstance(f_trues, list):
				for index, f_true in enumerate(f_trues):
					grid_z = griddata((xx, yy), f_true(xtest)[:, 0].numpy(), (grid_x, grid_y), method='linear')
					if colors is not None:
						color = colors[index]
					ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.4, color=color)
			else:
				grid_z = griddata((xx, yy), f_trues(xtest)[:, 0].numpy(), (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z, alpha=0.4)

			if filename is not None:
				plt.xticks(fontsize=20, rotation=0)
				plt.yticks(fontsize=20, rotation=0)
				plt.savefig(filename, dpi=300)

	def visualize_function_contour(self, xtest, f_true, filename=None, levels=10, figsize=(15, 7), alpha = 1., colorbar = True):
		d = xtest.size()[1]
		if d == 1:
			pass
		elif d == 2:
			from scipy.interpolate import griddata
			xx = xtest[:, 0].numpy()
			yy = xtest[:, 1].numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			f = f_true(xtest)
			grid_z_f = griddata((xx, yy), f[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')

			fig, ax = plt.subplots(figsize=figsize)
			cs = ax.contourf(grid_x, grid_y, grid_z_f, levels=levels, alpha = alpha, cmap = 'hot')
			ax.contour(cs, colors='k', alpha = 1.)
			if colorbar:
				cbar = fig.colorbar(cs)
			# if self.x is not None:
			#	ax.scatter(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), c='r', s=100, marker="o")
			ax.grid(c='k', ls='-', alpha=0.1)

			if filename is not None:
				plt.xticks(fontsize=24, rotation=0)
				plt.yticks(fontsize=24, rotation=0)
				plt.savefig(filename, dpi=300)
			return fig, ax
	# plt.show()

	def visualize(self, xtest, f_true=None, points=True, show=True, size=2,
				  norm=1, fig=True, sqrtbeta=2, constrained=None, d=None, matheron_kernel=None, color = None):

		[mu, std] = self.mean_std(xtest)

		if d is None:
			d = self.d

		if d == 1:
			if fig == True:
				plt.figure(figsize=(15, 7))
				plt.clf()
			if self.x is not None:
				plt.plot(self.x.detach().numpy(), self.y.detach().numpy(), 'ro', ms=10)
			if size > 0:

				if matheron_kernel is not None:
					z = self.sample_matheron(xtest, matheron_kernel, size=size).numpy().T
				else:
					z = self.sample(xtest, size=size).numpy().T

				for z_arr, label in zip(z, ['sample'] + [None for _ in range(size - 1)]):
					plt.plot(xtest.view(-1).numpy(), z_arr, 'k--', lw=2, label=label)

			plt.fill_between(xtest.numpy().flat, (mu - sqrtbeta * std).numpy().flat, (mu + sqrtbeta * std).numpy().flat,
							 color="#dddddd")
			if f_true is not None:
				plt.plot(xtest.numpy(), f_true(xtest).numpy(), 'b-', lw=2, label="truth")
			if color is None:
				plt.plot(xtest.numpy(), mu.numpy(), 'r-', lw=2, label="posterior mean")
			else:
				plt.plot(xtest.numpy(), mu.numpy(), linestyle = '-', lw=2, label="posterior mean", color = color)
			# plt.title('Posterior mean prediction plus 2 st.deviation')
			plt.legend()
			if show == True:
				plt.show()

		elif d == 2:
			from scipy.interpolate import griddata
			plt.figure(figsize=(15, 7))
			plt.clf()
			ax = plt.axes(projection='3d')
			xx = xtest[:, 0].numpy()
			yy = xtest[:, 1].numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			if f_true is not None:
				grid_z = griddata((xx, yy), f_true(xtest)[:, 0].numpy(), (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z, color='b', alpha=0.4, label="truth")
			if points == True and self.fit == True:
				ax.scatter(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), self.y[:, 0].detach().numpy(),
						   c='r', s=100, marker="o", depthshade=False)
			if self.beta is not None:
				beta = self.beta(norm=norm)
				grid_z2 = griddata((xx, yy), (mu.detach() + beta * std.detach())[:, 0].detach().numpy(),
								   (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z2, color='gray', alpha=0.2)
				grid_z3 = griddata((xx, yy), (mu.detach() - beta * std.detach())[:, 0].detach().numpy(),
								   (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z3, color='gray', alpha=0.2)

			ax.plot_surface(grid_x, grid_y, grid_z_mu, color='r', alpha=0.4)
			# plt.title('Posterior mean prediction plus 2 st.deviation')
			plt.show()

		else:
			print("Visualization not implemented")

	def visualize_subopt(self, xtest, f_true=None, points=True, show=True, size=2, norm=1, fig=True, beta=2):
		[mu, std] = self.mean_std(xtest)

		print("Visualizing in: ", self.d, "dimensions...")

		if self.d == 1:
			if fig == True:
				plt.figure(figsize=(15, 7))
				plt.clf()
			if self.x is not None:
				plt.plot(self.x.detach().numpy(), self.y.detach().numpy(), 'r+', ms=10, marker="o")
			plt.plot(xtest.numpy(), self.sample(xtest, size=size).numpy(), 'k--', lw=2, label="sample")
			plt.fill_between(xtest.numpy().flat, (mu - 2 * std).numpy().flat, (mu + 2 * std).numpy().flat,
							 color="#dddddd")
			if f_true is not None:
				plt.plot(xtest.numpy(), f_true(xtest).numpy(), 'b-', lw=2, label="truth")
			plt.plot(xtest.numpy(), mu.numpy(), 'r-', lw=2, label="posterior mean")

			min = torch.max(mu - beta * std)
			mask = (mu + beta * std < min)
			v = torch.min(mu - beta * std).numpy() - 1
			plt.plot(xtest.numpy()[mask], 0 * xtest.numpy()[mask] + v, 'ko', lw=6, label="Discarted Region")

			plt.title('Posterior mean prediction plus 2 st.deviation')
			plt.legend()

			if show == True:
				plt.show()

	def visualize_slice(self, xtest, slice, show=True, eps=None, size=1, beta=2):
		append = torch.ones(size=(xtest.size()[0], 1), dtype=torch.float64) * slice
		xtest2 = torch.cat((xtest, append), dim=1)

		[mu, std] = self.mean_std(xtest2)

		plt.figure(figsize=(15, 7))
		plt.clf()
		plt.plot(xtest.numpy(), self.sample(xtest, size=size).numpy(), 'k--', lw=2, label="sample")
		print(std.size(), mu.size())
		if self.x is not None:
			plt.plot(self.x[:, 0].detach().numpy(), self.y.detach().numpy(), 'r+', ms=10, marker="o")
		plt.fill_between(xtest.numpy().flat, (mu - 2 * std).numpy().flat, (mu + 2 * std).numpy().flat, color="#dddddd")
		plt.fill_between(xtest.numpy().flat, (mu + 2 * std).numpy().flat, (mu + 2 * std + 2 * self.s).numpy().flat,
						 color="#bbdefb")
		plt.fill_between(xtest.numpy().flat, (mu - 2 * std - 2 * self.s).numpy().flat, (mu - 2 * std).numpy().flat,
						 color="#bbdefb")

		if eps is not None:
			mask = (beta * std < eps)
			v = torch.min(mu - beta * std - 2 * self.s).numpy()
			plt.plot(xtest.numpy()[mask], 0 * xtest.numpy()[mask] + v, 'k', lw=6,
					 label="$\\mathcal{D}_E$ - $\\epsilon$ accurate domain in a subspace")

		plt.plot(xtest.numpy(), mu.numpy(), 'r-', lw=2, label="posterior mean")
		plt.title('Posterior mean prediction plus 2 st.deviation')
		plt.legend()
		if show == True:
			plt.show()

	def visualize_contour_with_gap(self, xtest, f_true=None, gap=None, show=False):
		[mu, _] = self.mean_std(xtest)

		if self.d == 2:
			from scipy.interpolate import griddata
			xx = xtest[:, 0].detach().numpy()
			yy = xtest[:, 1].detach().numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')

			fig, ax = plt.subplots(figsize=(15, 7))
			cs = ax.contourf(grid_x, grid_y, grid_z_mu)
			ax.contour(cs, colors='k')

			ax.plot(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), 'ro', ms=10)
			cbar = fig.colorbar(cs)

			ax.grid(c='k', ls='-', alpha=0.1)

			if f_true is not None:
				f = f_true(xtest)
				grid_z_f = griddata((xx, yy), f[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
				fig, ax = plt.subplots(figsize=(15, 7))
				cs = ax.contourf(grid_x, grid_y, grid_z_f)
				ax.contour(cs, colors='k')
				cbar = fig.colorbar(cs)
				ax.grid(c='k', ls='-', alpha=0.1)
			if show == True:
				plt.show()

	def visualize_contour(self, xtest, f_true=None, show=True, points=True, ms=5, levels=20):
		[mu, _] = self.mean_std(xtest)

		if self.d == 2:
			from scipy.interpolate import griddata
			xx = xtest[:, 0].detach().numpy()
			yy = xtest[:, 1].detach().numpy()

			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')

			fig, ax = plt.subplots(figsize=(15, 7))
			cs = ax.contourf(grid_x, grid_y, grid_z_mu)
			ax.contour(cs, colors='k')

			if points == True:
				ax.plot(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), 'wo', ms=ms, alpha=0.5)
			cbar = fig.colorbar(cs)
			ax.grid(c='k', ls='-', alpha=0.1)

			if f_true is not None:
				f = f_true(xtest)
				grid_z_f = griddata((xx, yy), f[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
				fig, ax = plt.subplots(figsize=(15, 7))
				cs = ax.contourf(grid_x, grid_y, grid_z_f, levels=levels)
				ax.contour(cs, colors='k')
				cbar = fig.colorbar(cs)
				ax.grid(c='k', ls='-', alpha=0.1)
			if show == True:
				plt.show()
			return ax

	def visualize_quiver(self, xtest, size=2, norm=1):
		[mu, std] = self.mean_std(xtest)
		if self.d == 2:
			from scipy.interpolate import griddata
			plt.figure(figsize=(15, 7))
			plt.clf()
			ax = plt.axes(projection='3d')
			xx = xtest[:, 0].detach().numpy()
			yy = xtest[:, 1].detach().numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			#

			ax.scatter(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), self.y[:, 0].detach().numpy(),
					   c='r', s=100, marker="o", depthshade=False)

			if self.beta is not None:
				beta = self.beta(norm=norm)
				grid_z2 = griddata((xx, yy), (mu.detach() + beta * std.detach())[:, 0].detach().numpy(),
								   (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z2, color='gray', alpha=0.2)
				grid_z3 = griddata((xx, yy), (mu.detach() - beta * std.detach())[:, 0].detach().numpy(),
								   (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z3, color='gray', alpha=0.2)

			ax.plot_surface(grid_x, grid_y, grid_z_mu, color='r', alpha=0.4)
			plt.title('Posterior mean prediction plus 2 st.deviation')

			derivatives = torch.zeros(xtest.size()[0], 2)
			for index, point in enumerate(xtest):
				derivatives[index, :] = self.mean_gradient_hessian(point.view(-1, 2))
				print(derivatives[index, :])

			print(derivatives.size())

			grid_der_x_mu = griddata((xx, yy), derivatives[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			grid_der_y_mu = griddata((xx, yy), derivatives[:, 1].detach().numpy(), (grid_x, grid_y), method='linear')

			fig, ax = plt.subplots(figsize=(15, 7))
			cs = ax.contourf(grid_x, grid_y, grid_z_mu)

			ax.contour(cs, colors='k')

			# Plot grid.
			ax.grid(c='k', ls='-', alpha=0.1)
			ax.quiver(grid_x, grid_y, grid_der_x_mu, grid_der_y_mu)

			plt.show()

		else:
			print("Visualization not implemented")


if __name__ == "__main__":
	from stpy.continuous_processes.kernelized_features import KernelizedFeatures
	from stpy.kernels import KernelFunction
	from stpy.embeddings.embedding import HermiteEmbedding
	import stpy
	import torch
	import matplotlib.pyplot as plt
	import numpy as np

	n = 1024
	N = 256
	gamma = 0.09
	s = 0.1
	# benchmark = stpy.test_functions.benchmarks.GaussianProcessSample(d =1, gamma = gamma, sigma = s, n = n)
	benchmark = stpy.test_functions.benchmarks.Simple1DFunction(d=1, sigma=s)

	x = benchmark.initial_guess(N, adv_inv=True)
	y = benchmark.eval(x)
	xtest = benchmark.interval(1024)

	# GP = GaussianProcess(gamma=gamma, s=s)
	# GP.fit_gp(x, y)
	# GP.visualize(xtest, show=False, size=5)
	# plt.show()

	m = 64
	kernel = KernelFunction(gamma=gamma)
	embedding = HermiteEmbedding(gamma=gamma, m=m)
	RFF = KernelizedFeatures(embedding=embedding, s=s, m=m)
	RFF.fit_gp(x, y)
	RFF.visualize(xtest, fig=False, show=False, size=5, matheron_kernel=kernel)
	plt.show()
