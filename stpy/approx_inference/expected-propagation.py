import scipy.integrate as integrate
import torch
from scipy.stats import multivariate_normal


class ExpectedPropagationQuadratic():

	def __init__(self, mu_prior, Sigma_prior, likelihood_single, data):

		# takes two arguments param, theta
		self.likelihood_single = likelihood_single

		# prior information
		self.mu_prior = mu_prior
		self.Sigma_prior = Sigma_prior

		self.d = mu_prior.size()[1]

		self.n = len(self.data)
		self.data = data

		self.approx = []
		for i in range(self.n):
			mu = torch.zeros(size=(1, self.d)).double()
			Sigma = torch.eye(size=(self.d, self.d)).double()
			self.approx.append((mu, Sigma))

	def marginalized_version(self, j):
		mu = torch.zeros(size=(1, self.d)).double()
		Sigma = torch.zeros(size=(self.d, self.d)).double()

		for i in range(self.n):
			if i != j:
				Sigma_elem = self.approx[j][0]
				mu_elem = self.approx[j][1]
				Sigma_elem_inv = torch.inverse(Sigma_elem)
				mu += Sigma_elem_inv @ mu_elem
				Sigma += Sigma_elem_inv
		Sigma = torch.inverse(Sigma)
		mu = Sigma @ mu
		return (mu, Sigma)

	def match_likelihood(self, j):
		mu, Sigma = self.marginalized_version(j)
		lik = lambda x: self.likelihood_single(torch.from_numpy(x), self.data[j]).numpy()
		prob = lambda x: multivariate_normal.pdf(x, mean=mu.view(-1).reshape.numpy(), cov=Sigma.numpy())
		first_moment = integrate.quad(lambda x: x * lik(x) * prob(x), 0.0, 10e10)
		second_moment = integrate.quad(lambda x: x * x * lik(x) * prob(x), 0.0, 10e10)

		self.approx[j][0] = first_moment
		self.approx[j][1] = second_moment

		return (first_moment, second_moment - first_moment ** 2)

	def finalize(self):
		pass

	def fit_gp(self, iterations='auto'):
		if iterations == 'auto':
			T = 100
		for i in range(T):
			for j in range(self.n):
				self.match_likelihood(j)
		mu, Sigma = self.finalize()
		return mu, Sigma
