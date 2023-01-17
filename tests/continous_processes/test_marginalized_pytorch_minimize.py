import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.helpers.helper import interval

#%%

n = 100
d = 2
x = torch.rand(n,d).double()*2 - 1
xtest = torch.from_numpy(interval(50,2,L_infinity_ball=1))

#%%

GP = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=2)
y = GP.sample(x)
GP.fit_gp(x,y)
GP.visualize_contour(xtest, ms = 10)

#%%

## Kernels can be defined as via kernel object
# 2 dimensional additive kernel with groups [0] and [1]
k = KernelFunction(kernel_name = "ard", d = 2, groups = [[0,1]] )
GP = GaussianProcess(kernel=k)

GP.fit_gp(x,y)
GP.optimize_params(type="bandwidth", restarts = 2, verbose = False, optimizer = 'pytorch-minimize')
GP.visualize_contour(xtest, ms = 10)
