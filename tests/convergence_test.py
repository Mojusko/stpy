from stpy.helpers.helper import *
from stpy.embeddings.embedding import *
from stpy.continuous_processes.fourier_fea import GaussianProcessFF
from stpy.continuous_processes.gauss_procc import GaussianProcess
import torch

# domain size
L_infinity_ball = 1
# dimension
d = 1
# error variance
s = 0.001
# grid density
n = 50
# number of intial points
N = 3
# smoothness
gamma = torch.ones(d, dtype= torch.float64)*1
# test problem

xtest = torch.from_numpy(interval(n, d))
x = torch.from_numpy(np.random.uniform(-L_infinity_ball, L_infinity_ball, size=(N, d)))

f_no_noise = lambda q: torch.sin(torch.sum(q * 4, dim=1)).view(-1, 1)
f = lambda q: f_no_noise(q) + torch.normal(mean=torch.zeros(q.size()[0], 1, dtype=torch.float64), std=1.,
										   out=None) * s
# targets
y = f(x)

# GP model with squared exponential
m = 12
groups = None
GP = GaussianProcess(kernel = "squared_exponential", s=s, gamma = gamma[0], d=d, groups = groups)
GP_KL = GaussianProcessFF(kernel="squared_exponential", s=s, m=m, d=d, gamma=gamma[0], groups=groups, approx="kl")
GP_He = GaussianProcessFF(kernel="squared_exponential", s=s, m=m, d=d, gamma=gamma[0], groups=groups, approx="hermite")

# fit GP
GP.fit_gp(x, y)
GP_KL.fit_gp(x, y)
GP_He.fit_gp(x, y)

print (GP.K)
print (GP_KL.right_kernel())
print (GP_He.right_kernel())