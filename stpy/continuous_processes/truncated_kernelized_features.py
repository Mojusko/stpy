from stpy.continuous_processes.kernelized_features import KernelizedFeatures
import torch 

class TruncatedKernelizedFeatures(KernelizedFeatures):

	def __init__(self, embedding, m, s=0.001, lam=1, d=1, diameter=1, verbose=True, groups=None, bounds=None, scale=1, kappa=1, poly=2,
	 primal=True, beta_fun=None, alpha_score=lambda t: t**(1/4), default_alpha_score=1.):
		super().__init__(embedding, m, s, lam, d, diameter, verbose,
			  groups, bounds, scale, kappa, poly, primal, beta_fun)
		primal = True
		self.alpha_score = alpha_score
		self.default_aplha_score = default_alpha_score

	def theta_mean(self, var=False, prior=False):
		self.precompute()

		if self.fit == True and prior == False:
			theta_mean = self.invV@self.Q.T@self.y_truncated
			Z = self.s**2 * self.invV
		else:
			theta_mean = 0*torch.ones(size=(self.m, 1)).double()

		if var is False:
			return theta_mean
		else:
			return (theta_mean, Z)

	def fit_gp(self, x, y):
		super().fit_gp(x, y)
		self.alphas = self.y*0 + self.default_aplha_score

	def add_points(self,x, y):
		if self.x is not None:
			self.x = torch.cat((self.x, x), dim=0)
			self.y = torch.cat((self.y, y), dim=0)
			new_alpha =torch.Tensor( [self.alpha_score(self.x.size()[0])]).view(1,1)
			self.alphas = torch.cat((self.alphas,new_alpha),dim=0)
		else:
			self.x = x
			self.y = y
			self.alphas = self.default_alpha_score
		self.fit = False

	def add_data_point(self,x,y):
		self.add_points(x,y)
		
	def precompute(self):
		if self.fit == False:
			self.Q = self.embed(self.x)
			I = torch.eye(int(self.m)).double()
			Z_ = self.Q.T@self.Q
			self.V = Z_ + self.s **2 * self.lam *I
			self.invV = torch.pinverse(self.V)
			self.y_truncated = self.y.view(-1)*(torch.abs(self.y) < self.alphas).view(-1).double()
			self.y_truncated = self.y_truncated.view(-1,1)
			self.fit = True
		else:
			pass

	