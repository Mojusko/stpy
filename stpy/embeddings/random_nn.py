import numpy as np
import torch
import torch.nn as nn


class RandomMap(nn.Module):

	def __init__(self, d, m, fun, output=2):
		super(RandomMap, self).__init__()
		self.W = torch.normal(mean=torch.zeros(m, d, dtype=torch.float64), std=1. / np.sqrt(d * m) ** 2)
		self.W.requires_grad_(True)
		self.w = torch.normal(mean=torch.zeros(m, output, dtype=torch.float64), std=1. / np.sqrt(d * m) ** 2)
		self.w.requires_grad_(True)
		self.b = torch.normal(mean=torch.zeros(output, dtype=torch.float64), std=1. / np.sqrt(d * m) ** 2)
		self.b.requires_grad_(True)
		self.fun = fun
		self.output = output

	def map(self, x):
		y = self.fun(torch.mm(self.W, torch.t(x)))
		return y

	def forward(self, x):
		z = self.map(x)
		z = torch.mm(torch.t(z), self.w)
		return z

	def get_params(self):
		return [self.W, self.w]

	def get_params_last(self):
		return [self.w]

	def fit_map(self, x, y, epochs=1000, verbose=False, reg=0.1, lr=0.1):
		criterion = nn.MSELoss()

		import torch.optim as optim
		optimizer = optim.SGD([self.W, self.w], lr=lr)

		batch_size = 100

		for i in range(epochs):
			for j in range(x.size()[0] // batch_size):
				optimizer.zero_grad()  # zero the gradient buffers
				output = self.forward(x[j * batch_size:(j + 1) * batch_size])
				loss = criterion(output, y[j * batch_size:(j + 1) * batch_size])
				loss.backward(retain_graph=True)
				optimizer.step()  # Does the update

			if verbose == True or i % verbose == 0:
				output = self.forward(x)
				loss_full = criterion(output, y)
				print(i, loss_full)
				optimizer.step()  # Does the update

	def fit_map_lasso(self, x, y, epochs=1000, verbose=False, reg=0.1, lr=0.1, l1=0.1):
		criterion = nn.MSELoss()

		import torch.optim as optim
		optimizer = optim.SGD([self.W, self.w], lr=lr)

		batch_size = 100

		for i in range(epochs):
			for j in range(x.size()[0] // batch_size):
				optimizer.zero_grad()  # zero the gradient buffers
				output = self.forward(x[j * batch_size:(j + 1) * batch_size])
				loss = criterion(output, y[j * batch_size:(j + 1) * batch_size]) + l1 * torch.norm(self.W, 2)
				loss.backward(retain_graph=True)
				optimizer.step()  # Does the update

			if verbose == True or i % verbose == 0:
				output = self.forward(x)
				loss_full = criterion(output, y)
				print(i, loss_full)
				optimizer.step()  # Does the update

	def loss(self, x, y):
		criterion = nn.MSELoss()
		output = self.forward(x)
		loss = criterion(output, y)

		return loss

	def fit_last_layer(self):
		# same as before but different parameters
		pass


class SpecificMap(RandomMap):

	def __init__(self, d, m, fun, map, output=2):
		super(SpecificMap, self).__init__(d, m, fun, output=2)
		self.map = map

	def forward(self, x):
		z = self.map(x)
		z = torch.mm(torch.t(z), self.w)
		return z

	def get_params(self):
		return [self.w]


def RandomMapStacked(RandomMap):
	def __init__(self, d, m, fun, output=2):
		super(RandomMap, self).__init__()
		self.W = torch.normal(mean=torch.zeros(m, d, dtype=torch.float64), std=1. / np.sqrt(d * m) ** 2)
		self.W.requires_grad_(True)
		self.w = torch.normal(mean=torch.zeros(m, output, dtype=torch.float64), std=1. / np.sqrt(d * m) ** 2)
		self.w.requires_grad_(True)
		self.b = torch.normal(mean=torch.zeros(m, 1, dtype=torch.float64), std=1. / np.sqrt(d * m) ** 2)
		self.b.requires_grad_(True)
		self.fun = fun
		self.output = output

	def map(self, x):
		y = self.fun(torch.mm(self.W, torch.t(x)) + self.b)
		return y

	def fit_map(self, x, y):
		pass


class RandomOrthogonalMap(RandomMap):

	def __init__(self, d, m, fun, output=1):
		super(RandomMap, self).__init__()
		self.m = m

		self.R = torch.normal(mean=torch.zeros(m, d, dtype=torch.float64), std=1. / np.sqrt(d * m) ** 2)
		self.R = nn.init.orthogonal_(self.R)
		self.R.requires_grad_(True)

		self.w = torch.normal(mean=torch.zeros(m, output, dtype=torch.float64), std=1. / np.sqrt(d * m) ** 2)
		self.w.requires_grad_(True)

		self.fun = fun
		self.output = output

	def map(self, x):
		y = self.fun(torch.mm(self.R, torch.t(x)))
		return y

	def fit_map(self, x, y, epochs=1000, verbose=False, reg=0.1, lr=0.1):
		criterion = nn.MSELoss()

		import torch.optim as optim

		optimizer = optim.SGD([self.R, self.w], lr=lr)
		orth_loss = torch.norm(torch.mm(self.R, torch.t(self.R)) - torch.eye(self.m, self.m, dtype=torch.float64)) ** 2

		batch_size = 100

		for i in range(epochs):
			for j in range(x.size()[0] // batch_size):
				optimizer.zero_grad()  # zero the gradient buffers
				output = self.forward(x[j * batch_size:(j + 1) * batch_size])
				loss = criterion(output, y[j * batch_size:(j + 1) * batch_size]) + reg * orth_loss
				loss.backward(retain_graph=True)
				optimizer.step()  # Does the update

			if verbose == True or i % verbose == 0:
				output = self.forward(x)
				loss_full = criterion(output, y) + reg * orth_loss
				print(i, loss_full)


class RandomNestedMap():

	def __init__(self):
		pass


if __name__ == "__main__":
	ridge = lambda x: torch.tanh(x)

	N = 1000
	d = 10
	m = 2

	NetOriginal = RandomMap(d, m, ridge)

	x = 10 * torch.normal(mean=torch.zeros(N, d, dtype=torch.float64) + 2, std=100.)
	y = NetOriginal.forward(x)

	Net = RandomMap(d, m, ridge)
	Net.fit_map(x, y)
