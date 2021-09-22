from stpy.helpers.helper import *
from stpy.cost_functions import CostFunction
from pymanopt.manifolds import Euclidean, Rotations, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from scipy.optimize import minimize


class HyperParameterOpt:

	def __init__(self, obj, x, y, fun, params):

		self.mode = obj
		self.x = x
		self.y = y
		self.fun = fun
		self.params = params

	def optimize(self, type, optimizer, restarts):


		## Bandwidth optimization
		def bandwidth_opt(X):
			gamma = X
			Rot = torch.eye(self.x.size()[1], dtype=torch.float64)
			return self.log_marginal_likelihood(gamma,Rot, 1.0, kernel=" ")


		def bandwidth_opt_handler():
			manifold = Euclidean(self.kernel_object.gamma.size()[0])
			C = CostFunction(bandwidth_opt, number_args = 1)
			xinit = lambda : np.random.randn()**2+np.abs(torch.zeros(self.kernel_object.gamma.size()[0], dtype = torch.float64).numpy())
			return optimize(manifold, C, 1, xinit)


		def bandwidth_kappa_opt(X):
			gamma = X[0]
			kappa = X[1]
			Rot = torch.eye(self.x.size()[1], dtype=torch.float64)
			return self.log_marginal_likelihood(gamma,Rot,kappa, kernel=" ")


		def bandwidth_kappa_opt_handler():
			manifold1 = Euclidean(self.kernel_object.gamma.size()[0])
			manifold2 = Euclidean(1)
			manifold = Product((manifold1, manifold2))
			C = CostFunction(bandwidth_kappa_opt, number_args = 2)
			xinit = lambda x: [torch.randn(self.kernel_object.gamma.size()[0], dtype = torch.float64).numpy(),np.abs(torch.randn(1,dtype = torch.float64).numpy())]
			return optimize(manifold, C, 2, xinit)


		## Rotations optimization
		def rotations_opt(X):
			Rot = X
			return self.log_marginal_likelihood(self.kernel_object.gamma,Rot, self.kernel_object.kappa, kernel=" ")

		def rotations_opt_handler():
			rots = Rotations(self.kernel_object.gamma.size()[0])
			manifold = rots
			xinit = lambda : torch.qr(torch.randn(self.x.size()[1],self.x.size()[1], dtype=torch.float64))[0].numpy()
			C = CostFunction(rotations_opt, number_args = 1)
			return optimize(manifold, C, 1, xinit)



		## Bandwidth and Rotations optimization
		def bandwith_rotations_opt(X):
			gamma = X[0]
			Rot = X[1]
			return self.log_marginal_likelihood(gamma,Rot, 0.1, kernel=" ")

		def bandwidth_rotations_opt_handler():
			eucl = Euclidean(self.kernel_object.gamma.size()[0])
			rots = Rotations(self.kernel_object.gamma.size()[0])
			manifold = Product((eucl, rots))
			xinit = lambda : [torch.randn(self.kernel_object.gamma.size()[0], dtype = torch.float64).numpy(),torch.qr(torch.randn(self.x.size()[1],self.x.size()[1], dtype=torch.float64))[0].numpy()]
			C = CostFunction(bandwith_rotations_opt, number_args = 2)
			return optimize(manifold, C, 2, xinit)

		## Bandwidth and Rotations optimization
		def bandwith_kappa_rotations_opt(X):
			gamma = X[0]
			kappa = X[1]
			Rot = X[2]
			return self.log_marginal_likelihood(gamma,Rot, kappa, kernel=" ")

		def bandwidth_kappa_rotations_opt_handler():
			eucl = Euclidean(self.kernel_object.gamma.size()[0])
			eucl2 = Euclidean(1)
			rots = Rotations(self.kernel_object.gamma.size()[0])
			manifold = Product((eucl, eucl2, rots))
			xinit = [self.kernel_object.gamma.numpy(),torch.eye(self.x.size()[1], dtype=torch.float64).numpy()]
			C = CostFunction(bandwith_kappa_rotations_opt, number_args = 2)
			return optimize(manifold, C, 2, xinit)




		# Finalize
		if type == "bandwidth":
			best_params = bandwidth_opt_handler()
			self.kernel_object.gamma = torch.abs(best_params[0]).detach()

		elif type == "rots":
			best_params = rotations_opt_handler()
			Rot = best_params[0].detach()
			print("Rotation:",Rot)
			self.Rot = Rot
			self.x = torch.mm(self.x,Rot).detach()

		elif type == "bandwidth+kappa":
			best_params = bandwidth_kappa_opt_handler()
			self.kernel_object.gamma = torch.abs(best_params[0]).detach()
			self.s = torch.abs(best_params[1]).detach()

		elif type == "bandwidth+rots":
			best_params = bandwidth_rotations_opt_handler()
			self.kernel_object.gamma = torch.abs(best_params[0]).detach()
			Rot = best_params[1].detach()
			print("Rotation:",Rot)
			self.Rot = Rot
			self.x = torch.mm(self.x,Rot).detach()

		elif type == "bandwidth+kappa+rots":
			best_params = bandwidth_kappa_rotations_opt_handler()
			self.kernel_object.gamma = torch.abs(best_params[0]).detach()
			self.s = torch.abs(best_params[1]).detach()
			Rot = best_params[2].detach()
			print("Rotation:", Rot)
			self.Rot = Rot
			self.x = torch.mm(self.x, Rot).detach()

		else:
			raise AttributeError("Optimization scheme not implemented")

		self.back_prop = False
		self.fit = False
		self.fit_gp(self.x,self.y)
		print (self.description())

		return True