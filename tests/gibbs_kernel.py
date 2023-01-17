from stpy.kernels import KernelFunction
from stpy.continuous_processes.gauss_procc import  GaussianProcess
from stpy.helpers.helper import interval
from stpy.embeddings.optimal_positive_basis import OptimalPositiveBasis
import matplotlib.pyplot as plt
import torch
import numpy as np
n = 1024
d = 1


def gamma(x):
	out = x[:,0].view(-1,1)*0
	small = x <= - 0.5
	mid = torch.logical_and(x >= -0.5,x <= 0.5)
	large = x > 0.5
	gamma1 = 0.1
	gamma2 = 1.
	out[small] = (gamma2-gamma1)/(torch.exp(-25*(x[small]+0.5)) + 1) + gamma1
	out[mid] = gamma2
	out[large] = (gamma2-gamma1)/(torch.exp(-25*(-x[large]+0.5)) + 1) + gamma1
	return out

gamma = lambda x: x[:,0].view(-1,1)*0 + 0.05 + 0.3*(x+1)**4

#gamma = lambda x: x[x<-0.5]*0 +0 + 0.05 + 0.2*(x+1)**2#*torch.abs(torch.cos(x*np.pi)) + 0.5
xtest = torch.from_numpy(interval(n,d))

vals = gamma(xtest).T**2 + gamma(xtest)**2
plt.imshow(vals)
plt.colorbar()
plt.show()

k = KernelFunction(kernel_name="gibbs", params={'gamma_fun':gamma})
plt.imshow(k.kernel(xtest,xtest))
plt.colorbar()
plt.show()

GP = GaussianProcess(kernel=k)

d = 1
m = 8
n = 256
sqrtbeta = 2
s = 0.01
b = 0

Emb = OptimalPositiveBasis(d, m, offset=0.0, s=s, b=b, discretization_size=n, B=1000., kernel_object=k, samples = 1000)
for i in range(m):
	f_i = Emb.basis_fun(xtest, i)  ## basis function
	plt.plot(xtest,f_i)

plt.show()

# ytest = GP.sample(xtest)
# plt.plot(xtest,ytest)
# plt.show()