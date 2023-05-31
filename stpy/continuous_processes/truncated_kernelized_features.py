from stpy.continuous_processes.kernelized_features import KernelizedFeatures
import torch 

class TruncatedKernelizedFeatures(KernelizedFeatures):

	def __init__(self, embedding, m, s=0.001, lam=1, d=1, diameter=1, verbose=True, groups=None, bounds=None, scale=1, kappa=1, poly=2,
	 primal=True, beta_fun=None, alpha_score=lambda t: t**(1/4), default_alpha_score=1., bound = 1.):
		super().__init__(embedding, m, s =s, lam=lam,d= d,diameter= diameter, verbose=verbose,
			  groups = groups, bounds=bounds, scale=scale, kappa=kappa, poly=poly, primal=primal, beta_fun = beta_fun, bound = bound)
		primal = True
		self.bound = bound
		self.alpha_score = alpha_score
		self.default_alpha_score = default_alpha_score

	def theta_mean(self, var=False, prior=False):
		self.precompute()

		if self.fitted == True and prior == False:
			theta_mean = self.invV@self.Q.T@self.y_truncated
			Z = self.s**2 * self.invV
		else:
			theta_mean = 0*torch.ones(size=(self.m, 1)).double()

		if var is False:
			return theta_mean
		else:
			return (theta_mean, Z)

	def fit(self, x=None, y=None):
		self.alphas = self.y*0 + self.default_alpha_score
		super().fit(x= x, y= y)

	def add_points(self,d):
		x, y = d
		if self.x is not None:
			self.x = torch.cat((self.x, x), dim=0)
			self.y = torch.cat((self.y, y), dim=0)
			new_alpha =torch.Tensor( [self.alpha_score(self.x.size()[0])]).view(1,1)
			self.alphas = torch.cat((self.alphas,new_alpha),dim=0)
		else:
			self.x = x
			self.y = y
			self.alphas = self.default_alpha_score
		self.fitted = False

	def add_data_point(self,x,y):
		self.add_points(x,y)
		
	def precompute(self):
		if self.fitted == False:
			self.Q = self.embed(self.x)
			I = torch.eye(int(self.m)).double()
			Z_ = self.Q.T@self.Q
			self.V = Z_ + (self.s **2) * self.lam *I
			self.invV = torch.pinverse(self.V)
			self.y_truncated = self.y.view(-1)*(torch.abs(self.y) < self.alphas).view(-1).double()
			self.y_truncated = self.y_truncated.view(-1,1)
			self.fitted = True
		else:
			pass

