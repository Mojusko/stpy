import pickle

from h5py import File

from stpy.helpers.helper import *


class FelSimulator():

	def __init__(self, d, sigma, name):
		self.d = d
		self.sigma = sigma
		self.exp_name = name

	def help(self, reload=False):
		print("Help for the FelSimulator")

	def load_pickle(self, file_name):

		self.GP = pickle.load(open(file_name, "rb"))
		self.d = self.GP.d
		self.exp_name = self.GP.exp_name

	def save(self, file_name):
		self.GP.exp_name = self.exp_name
		pickle.dump(self.GP, open(file_name, "wb"), -1)

	def load_fresh(self, file_name, dts='1'):
		f = File(file_name, 'r')
		dset = f[dts]
		print(dset)
		n = dset[str("x")].shape[0]
		mask = np.full(n, False, dtype=bool)
		for j in range(self.d):
			maskNew = dset["line_id"] == j
			mask = np.logical_or(mask, maskNew)
		print("Using ", np.sum(mask), "points to fit the model.")
		self.x = dset["x"][mask, 0:self.d].reshape(-1, self.d)
		self.y = dset["y"][mask].reshape(-1, 1)
		# y response and scale, x scale to [-0.5,0.5]
		scale = np.max(np.abs(self.y))
		self.y = self.y / scale
		for j in range(self.d):
			a = np.min(self.x[:, j])
			b = np.max(self.x[:, j])
			self.x[:, j] = (self.x[:, j] / (b - a)) - 0.5 - a / (b - a)
		# noise structure
		self.s = np.max(dset["y_std"][mask] / scale)
		print("The noise level estimated to be:", self.s)
		self.x = torch.from_numpy(self.x)
		self.y = torch.from_numpy(self.y)

		f.close()

	def fit_simulator(self, GP, optimize="bandwidth", restarts=10):
		self.GP = GP
		self.GP.s = self.s
		self.GP.fit(self.x, self.y)
		print("Model fitted.")
		self.GP.optimize_params(type=optimize, restarts=restarts)
		self.GP.back_prop = True

	def bounds(self, N, n):
		x = torch.from_numpy(np.random.uniform(-0.5, 0.5, size=(N, self.GP.d)))
		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(-0.5, 0.5, n).reshape(n, 1) for i in range(self.GP.d)]
			xtest = cartesian(arrays)
			xtest = torch.from_numpy(xtest)
		return (x, xtest, self.GP.d, None)

	def opt_bounds(self):
		bounds = tuple([(-0.5, 0.5) for i in range(self.GP.d)])
		return bounds

	def constraint(self, X):
		return True

	def eval(self, X, sigma=None):
		if sigma is None:
			sigma = self.sigma
		[mu, _] = self.GP.mean_std(X)
		return mu + sigma * torch.randn(X.size()[0], 1, dtype=torch.float64)

	def eval_sample(self, X, sigma=None):
		if sigma is None:
			sigma = self.sigma
		f = self.GP.sample(X)
		self.x = torch.cat((self.x, X), dim=0)
		self.y = torch.cat((self.y, f), dim=0)
		self.GP.fit(self.x, self.y)
		return f

	def optimum(self):
		## find optimum using backpropagation optimize eval_sample given X
		x = torch.randn(self.d, 1, requires_grad=True)
		x0 = x

		from scipy.optimize import minimize

		def fun(x):
			x = np.array([x])
			return -self.eval(torch.from_numpy(x)).numpy()[0][0]

		def grad(x):
			z = torch.from_numpy(np.array([x]))
			z.requires_grad_(True)
			y = -self.eval(z)
			y.backward()
			return z.grad.numpy()[0]

		mybounds = self.opt_bounds()
		res = minimize(fun, x0.detach().numpy(), method="L-BFGS-B", jac=grad, tol=0.0001, bounds=mybounds)
		solution = res.x

		val = self.eval(torch.from_numpy(solution).unsqueeze(0))
		loc = torch.from_numpy(solution).unsqueeze(0)

		return (val, loc)
