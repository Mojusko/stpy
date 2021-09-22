import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
from stpy.continuous_processes.gauss_procc import GaussianProcess

class CategoricalMixture(GaussianProcess):

	def __init__(self,processes,init_weights = None,d = 1, bounds = None):
		if init_weights is None:
			self.k = len(processes)
			init_weights = torch.ones(size = (self.k,1)).view(-1).double()*1./float(self.k)
		else:
			self.k = len(processes)

		if len(processes) != init_weights.shape[0]:
			raise AssertionError("Not the same number")

		self.processes = processes
		self.bounds = bounds
		self.beta = 2.
		self.d = d
		self.x = None
		self.y = None
		self.init_weights = init_weights
		if torch.sum(self.init_weights)>1.:
			self.init_weights = self.init_weights/torch.sum(self.init_weights)
		self.weights = self.init_weights

	def add_data_point(self,x,y):
		for model in self.processes:
			model.add_data_point(x,y)

	def log_prob_normal(self,K,y):
		Knumpy = K.detach().numpy()
		ynumpy = y.detach().numpy()

		decomp = scipy.linalg.lu_factor(Knumpy)
		alpha = scipy.linalg.lu_solve(decomp, ynumpy)

		logprob = -0.5 * ynumpy.T.dot(alpha) - 0.5*np.linalg.slogdet(Knumpy)[1] - 0.5*ynumpy.shape[0]*np.log(2*np.pi)

		return float(logprob)

	def fit_gp(self, x, y, iterative = False):
		self.x = x
		self.y = y

		logprobs = torch.zeros(size=(self.k,1)).view(-1).double()

		for j in range(self.k):
			GP = self.processes[j]
			GP.fit_gp(x,y)
			K = GP.get_kernel()
			logprobs[j] = self.log_prob_normal(K,y)

		#print("Neg. log likelihood vector:", -logprobs)

		log_init_prob = torch.log(self.init_weights)
		log_posterior = log_init_prob + logprobs
		log_evidence = torch.logsumexp(log_posterior,dim = 0)
		self.weights = torch.exp(log_posterior - log_evidence)

		#print ("Categorical Probability: ",self.weights)
		#print ("---------------------------------")

		self.fit = True
		return True


	def mean_std(self, xtest):
		mu = torch.zeros(size = (xtest.size()[0],1)).double()
		s = torch.zeros(size = (xtest.size()[0],1)).double()
		for j in range(self.k):
			(a1,a2)= self.processes[j].mean_std(xtest)

			mu = mu +  self.weights[j]*a1
			s = s + self.weights[j]*a2**2
		s = torch.sqrt(s)
		return (mu,s)


	def sample(self,xtest, size = 1, with_mask = False):
		# sample a GP
		k = np.random.choice(np.arange(0,self.k,1), p = self.weights.flatten())
		mask = [k]
		if self.fit == True:
			self.processes[k].fit_gp(self.x, self.y)
			samples = self.processes[k].sample(xtest, size=1)
		else:
			samples = self.processes[k].sample(xtest, size=1)

		for s in range(size-1):
			k = np.random.choice(np.arange(0, self.k, 1), p=self.weights.flatten())
			mask.append(k)
			if self.fit == True:
				self.processes[k].fit_gp(self.x,self.y)
				sample = self.processes[k].sample(xtest,size = 1)
				samples = torch.cat((samples, sample), dim=1)
			else:
				sample = self.processes[k].sample(xtest, size = 1)
				samples = torch.cat((samples, sample), dim=1)
		if with_mask == True:
			return (samples,mask)
		else:
			return samples

		
if __name__ == "__main__":

	# domain size
	L_infinity_ball = 5
	# dimension
	d = 1
	# error variance
	s = 0.001
	# grid density
	n = 512
	# number of intial points
	N = 15

	# model
	#GP1 = GaussianProcess(kernel="squared_exponential", s=s, gamma = 1.5, diameter=L_infinity_ball)
	GP1 = GaussianProcess(kernel="modified_matern", s=s, kappa=1., nu=2, gamma=1.5)
	GP2 = GaussianProcess(kernel="modified_matern", s=s, kappa=1., nu=1, gamma=0.7)
	#GP2 = GaussianProcess(kernel="squared_exponential", s=s, gamma=1.1)
	GP3 = GaussianProcess(kernel = "modified_matern", s = s, kappa = 1., nu = 2, gamma = 1)
	GP4 = GaussianProcess(kernel="linear", s=s, kappa=1.)

	# data
	#GPTrue = GaussianProcess(kernel="linear", s=0, kappa=1., diameter=L_infinity_ball)
	#GPTrue = GaussianProcess(kernel="squared_exponential", s=s, gamma=2., kappa = 1)
	GPTrue = GaussianProcess(kernel = "modified_matern", s =s, kappa = 1., nu = 2, gamma = 1.1)

	# test environment

	d = 1
	from stpy.test_functions.benchmarks import GaussianProcessSample
	BenchmarkFunc = GaussianProcessSample(d = d, n = n, sigma = 0., gamma = 0.2, name = "squared_exponential")
	x = BenchmarkFunc.initial_guess(N)
	xtest = BenchmarkFunc.interval(n)
	BenchmarkFunc.optimize(xtest, s)
	gamma = BenchmarkFunc.bandwidth()
	bounds = BenchmarkFunc.bounds()
	BenchmarkFunc.scale_max(xtest = xtest)
	F = lambda x: BenchmarkFunc.eval(x,sigma = s)

	# targets
	y = F(x)
	GPs = [GP1,GP2,GP3,GP4]
	#Mix = CategoricalMixture(GPs,init_weights=np.array([0.01,0.01,0.98]))
	Mix = CategoricalMixture(GPs)


	for j in range(N):


		plt.figure(1)
		plt.clf()
		X = x[0:j+1,:].reshape(-1,1)
		y = F(X)
		Mix.fit_gp(X, y)
		(mu,var) = Mix.mean_std(xtest)
		samples = Mix.sample(xtest, size=5)
		f = F(xtest).numpy()
		mu = mu.numpy()
		var =var.numpy()
		samples = samples.numpy()
		xtest2 = xtest.numpy()

		plt.plot(xtest2, samples,'--', linewidth = 2, alpha = 0.3)
		plt.plot(xtest2, mu,'k', linewidth = 3)
		plt.plot(xtest2, mu, 'k', linewidth=3)
		plt.fill_between(xtest2.flat, (mu - 2*var).flat, (mu + 2*var).flat, color="#dddddd")
		plt.plot(X, y, 'ro', markersize=10)
		plt.plot(xtest2,f,'g',linewidth = 3)
		plt.draw()

		plt.figure(2)
		plt.clf()
		plt.title("Probability of Category")
		plt.bar(np.arange(len(GPs)), Mix.weights, np.ones(len(GPs))*0.5)
		plt.xticks(np.arange(len(GPs)), [GP.description() for GP in GPs], rotation=30)
		plt.subplots_adjust(bottom=0.35)
		plt.plot()
		plt.show()
		#plt.pause(4)


