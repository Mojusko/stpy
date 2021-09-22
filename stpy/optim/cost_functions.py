import numpy as np
import torch

class CostFunction:

	def __init__(self,cost,number_args = 1):
		self.cost = cost
		self.number_args = number_args

	def joined_egrad(self,Xx):
		for X in Xx:
			X.requires_grad_(True)
		y = self.cost(Xx)
		y.backward(retain_graph=True)
		output = []
		for X in Xx:
			output.append(X.grad)
		return output

	def joined_hess(self,Xx,Uu):
		for X in zip(Xx):
			X.requires_grad_(True)
		y = self.joined_egrad(Xx)
		y.backward(retain_graph=True)
		output = []
		for X,U in zip(Xx,Uu):
			output.append(torch.mm(X.grad,Uu))
		return output

	def egrad(self,X):
		X.requires_grad_(True)
		y = self.cost(X)
		y.backward(retain_graph=True)
		return X.grad

	def ehess(self,X,U):
		X.requires_grad_(True)
		y = self.egrad(X)
		y.backward(retain_graph=True)
		return torch.mm(X.grad,U)

	def define(self):
		if self.number_args == 1:
			cost_numpy = lambda X: self.cost(torch.from_numpy(X)).data.numpy()
			grad_numpy = lambda X: self.egrad(torch.from_numpy(X)).data.numpy()
			hess_numpy = lambda X,U: self.ehess(torch.from_numpy(X),torch.from_numpy(U)).data.numpy()
			return [cost_numpy, grad_numpy, hess_numpy]
		else:
			cost_numpy = lambda Xx: self.cost([torch.from_numpy(X) for X in Xx]).data.numpy()
			grad_numpy = lambda Xx: [z.data.numpy() for z in self.joined_egrad([torch.from_numpy(X) for X in Xx])]
			hess_numpy = lambda Xx, Uu: [z.data.numpy() for z in self.joined_ehess([torch.from_numpy(X) for X in Xx],[torch.from_numpy(U) for U in Uu])]
			return [cost_numpy, grad_numpy, hess_numpy]
