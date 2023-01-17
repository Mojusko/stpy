import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from stpy.helpers.helper import cartesian


class BorelSet():

	def __init__(self, d, bounds):
		self.d = d
		self.bounds = bounds
		self.calculate_volume()
		self.type = "box"

	def description(self):
		return self.bounds

	def calculate_volume(self):
		self.vol = 1.
		for i in range(self.d):
			self.vol = self.vol * (self.bounds[i, 1] - self.bounds[i, 0])

	def volume(self):
		return self.vol

	def center_point(self):
		return (self.bounds[:, 1] + self.bounds[:, 0]) / 2

	def perimeter(self):
		cir = 0.
		for i in range(self.d):
			cir += 2 * (self.bounds[i, 1] - self.bounds[i, 0])
		return cir

	def uniform_sample(self, n):
		sample = torch.zeros(n, self.d).double()
		for i in range(self.d):
			sample_i = torch.from_numpy(np.random.uniform(self.bounds[i, 0], self.bounds[i, 1], n))
			sample[:, i] = sample_i
		return sample

	def return_legendre_discretization(self, n):
		nodes, weights = np.polynomial.legendre.leggauss(n)
		nodes_arr = []
		weights_arr = []
		for i in range(self.d):
			a, b = float(self.bounds[i, 0]), float(self.bounds[i, 1])
			nodes = nodes * (b - a) / 2. + (a + b) / 2.
			nodes_arr.append(nodes)
			weights_arr.append(weights * 0.5 * (b - a))

		nodes = cartesian(nodes_arr)
		weights = cartesian(weights_arr)
		return torch.prod(torch.from_numpy(weights), dim=1), torch.from_numpy(nodes)

	def return_discretization(self, n, offsets=None):
		dis = []
		for i in range(self.d):
			if offsets is None:
				x = np.linspace(self.bounds[i, 0], self.bounds[i, 1], n)
			else:
				x = np.linspace(self.bounds[i, 0] - offsets[i], self.bounds[i, 1] + offsets[i], n)
			dis.append(x)
		r = cartesian(dis)
		r = torch.from_numpy(r)
		return r

	def inside(self, set):
		"""
		Tests if set is inside this set
		:param set:
		:return:
		"""
		for i in range(self.d):
			if self.bounds[i, 0] > set.bounds[i, 0] or self.bounds[i, 1] < set.bounds[i, 1]:
				return False
		return True

	def is_inside(self, x):
		"""
		:param x:  (n,d) to check if a<=x<b
		:return: bool
		"""
		mask = torch.full((x.size()[0], 1), True, dtype=torch.bool).view(-1)
		for i in range(self.d):
			mask1 = self.bounds[i, 0] <= x[:, i]
			mask2 = x[:, i] < self.bounds[i, 1]
			mask = mask1 * mask2 * mask
		return mask


class BallSet(BorelSet):
	def __init__(self, d, center, radius):
		self.d = d
		self.center = center
		self.radius = radius
		self.calculate_volume()
		self.type = "round"

	def calculate_volume(self):
		self.vol = (self.radius ** self.d) * np.pi ** (self.d // 2) / (scipy.special.gamma(self.d // 2 + 1))

	def description(self):
		return self.center, self.radius

	def return_discretization(self, n):
		if self.d == 1:
			dis = []
			x = np.linspace(self.center - self.radius, self.center + self.radius, n)
			dis.append(x)
			r = cartesian(dis)
			r = torch.from_numpy(r)
			return r


		elif self.d == 2:

			p, w = np.polynomial.legendre.leggauss(n)
			mu = np.arange(1, n + 1)
			points = np.array(
				[
					np.tile(self.radius * np.cos(mu * np.pi / (n + 1)), n),
					np.outer(p, self.radius * np.sin(mu * np.pi / (n + 1))).flatten(),
				]
			).T
			points[:, 0] += float(self.center[0])
			points[:, 1] += float(self.center[1])

		# k = n - 2
		# theta = 2 * np.pi * np.arange(1, k + 2) / (k + 1)
		# p, w = np.polynomial.legendre.leggauss(n + 1)
		# # scale points to [r0, r1] (where r0 = 0, r1 = 1 for now)
		# p = np.sqrt(0.5 * (p + 1.0))
		# p_theta = np.dstack(np.meshgrid(p, theta)).reshape(-1, 2).T
		# points = np.array(
		# 	[p_theta[0] * self.radius * np.cos(p_theta[1]), p_theta[0] * self.radius * np.sin(p_theta[1])]
		# ).T
		# points[:,0] += float(self.center[0])
		# points[:,1] += float(self.center[1])
		#
		# points = np.concatenate((points,self.center.view(-1,self.d).numpy()))

		return torch.from_numpy(points)

	def return_legendre_discretization(self, n):
		if self.d == 2:
			p, w = np.polynomial.legendre.leggauss(n)
			mu = np.arange(1, n + 1)
			points = np.array(
				[
					np.tile(self.radius * np.cos(mu * np.pi / (n + 1)), n),
					np.outer(p, self.radius * np.sin(mu * np.pi / (n + 1))).flatten(),
				]
			).T
			points[:, 0] += float(self.center[0])
			points[:, 1] += float(self.center[1])
			weights = np.outer(w, np.sin(mu * np.pi / (n + 1)) ** 2).flatten() / (n + 1)
			return torch.from_numpy(weights), torch.from_numpy(points)
		else:
			raise AssertionError("Wrong type of set considered.")

	def inside(self, set):
		"""
		Tests if set is inside this set
		:param set:
		:return:
		"""

		## the tested set is box
		if set.type == "box":
			for i in range(self.d):
				if set.bounds[i, 0] > self.center[i] - self.radius or set.bounds[i, 1] < self.center[i] - self.radius:
					return False
		## set is round
		else:
			if (self.center - set.center) ** 2 > self.radius ** 2:
				return False
		return True

	def is_inside(self, x):
		"""
		:param x:  (n,d) to check if a<=x<b
		:return: bool
		"""
		n = x.size()[0]
		mask = torch.full((x.size()[0], 1), True, dtype=torch.bool).view(-1)
		mask = self.radius ** 2 >= (x - torch.tile(self.center, (n, 1))) ** 2
		return mask


class Node(BorelSet):

	def __init__(self, d, bounds, parent):
		super().__init__(d, bounds)
		self.left = None
		self.right = None
		self.children = None
		self.parent = parent

		if self.parent is None:
			self.level = 1
		else:
			self.level = parent.level + 1


class HierarchicalBorelSets():

	def __init__(self, d, interval, levels):
		if d == 1:
			self.top_node = Node(d, torch.Tensor([interval]), None)
		elif d == 2:
			self.top_node = Node(d, torch.Tensor(interval), None)

		self.Sets = [self.top_node]
		self.levels = levels
		if d == 1:
			self.construct_1d(interval, levels, self.Sets, self.top_node)
		else:
			self.construct_2d(self.top_node.bounds, levels, self.Sets, self.top_node)
		self.d = d

	def get_parent_set(self):
		return self.top_node

	def get_sets_level(self, l):
		out = []
		for s in self.Sets:
			if s.level == l:
				out.append(s)
		return out

	def get_all_sets(self):
		return self.Sets

	def get_ball_coverings(self, n, radius='auto'):
		D = self.get_parent_set()
		centers = D.return_discretization(n)
		n = centers.size()[0]
		sets = []
		for i in range(n):
			if radius == 'auto':
				sets.append(BallSet(D.d, centers[i, :], 2. / n))
			else:
				sets.append(BallSet(D.d, centers[i, :], radius))
		return sets

	def construct_1d(self, interval, levels, S, parent):

		if levels > 1:
			a, b = interval
			c = (a + b) / 2.

			S_1 = Node(1, torch.Tensor([[a, c]]), parent)
			S_2 = Node(1, torch.Tensor([[c, b]]), parent)

			parent.left = S_1
			parent.right = S_2

			S.append(S_1)
			self.construct_1d((a, c), levels - 1, S, S_1)
			S.append(S_2)
			self.construct_1d((c, b), levels - 1, S, S_2)

		else:
			return None

	def construct_2d(self, interval, levels, S, parent):
		if levels > 1:
			xa = interval[0, 0]
			xb = interval[0, 1]
			ya = interval[1, 0]
			yb = interval[1, 1]

			midx = xa + (xb - xa) / 2.
			midy = ya + (yb - ya) / 2.

			S1 = Node(2, torch.Tensor([[xa, midx], [ya, midy]]), parent)
			S2 = Node(2, torch.Tensor([[xa, midx], [midy, yb]]), parent)
			S3 = Node(2, torch.Tensor([[midx, xb], [ya, midy]]), parent)
			S4 = Node(2, torch.Tensor([[midx, xb], [midy, yb]]), parent)

			parent.children = [S1, S2, S3, S4]

			for child in parent.children:
				S.append(child)
				self.construct_2d(child.bounds, levels - 1, S, child)
		else:
			return None


if __name__ == "__main__":
	center = torch.Tensor([0.5, 0.5]).double()
	radius = 0.1
	d = 2
	B = BallSet(d, center, radius)

	weights, xtest = B.return_legendre_discretization(10)
	xtest2 = B.return_discretization(10)
	print(torch.sum(weights))
	plt.plot(xtest[:, 0], xtest[:, 1], 'ko')
	plt.plot(xtest2[:, 0], xtest2[:, 1], 'ro')
	plt.show()
