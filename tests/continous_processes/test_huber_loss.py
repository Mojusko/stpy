import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.helpers.helper import interval
import matplotlib.pyplot as plt
import numpy as np

N = 10
n = 256
d = 1
eps = 0.01
s = 1
x = torch.rand(N,d).double()*2 - 1
xtest = torch.from_numpy(interval(n,d,L_infinity_ball=1))

# true
GP_true = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d)
ytest = GP_true.sample(xtest)
GP_true.fit_gp(xtest,ytest)

plt.plot(xtest,GP_true.mean(xtest),'b-')

y = GP_true.mean(x).clone()
GP = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d, loss = 'huber', huber_delta=0.01, s = s)

xnew = x[0,:].view(1,1) + eps
ynew = y[0,0].view(1,1) + 1

y2 = torch.vstack([y,ynew])
x2 = torch.vstack([x,xnew])

GP.fit_gp(x2,y2)

GP2 = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d)
GP3 = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d)

GP2.fit_gp(x2,y2)
#GP2.visualize(xtest, show = False, fig = False)
#plt.show()



### marginalized likelihood with normal loss_two_ways
# plot true function
plt.plot(xtest,GP_true.mean(xtest),'b--',label = "truth", lw = 3)

# with noise optimize
GP2.fit_gp(x2,y2)
GP2.optimize_params(type="bandwidth", restarts=5, verbose = False, optimizer = 'pytorch-minimize', scale = 1.)
mu = GP2.mean(xtest)
plt.plot(xtest,mu, 'r-', label = "squared-corupted", lw = 3)
#GP2.visualize(xtest, show = False, fig = False, size = 0)

# no noise optimize
GP2.fit_gp(x,y)
GP2.optimize_params(type="bandwidth", restarts=5, verbose = False, optimizer = 'pytorch-minimize', scale = 1.)
mu = GP2.mean(xtest)
plt.plot(xtest,mu, '--x', color ="tab:brown" , label = 'squared-uncorrupted', lw = 3)

# with huber optimize
GP = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d, loss = 'huber', huber_delta=1.3)
GP.fit_gp(x2,y2)
GP.optimize_params(type="bandwidth", restarts=5, verbose = False, optimizer = 'pytorch-minimize', scale = 1., weight=1.)
mu = GP2.mean(xtest)
plt.plot(xtest,mu, color = "tab:green", label = 'huber-corupted', lw = 3)

# GP = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d, loss = 'huber', huber_delta=1.3)
# GP.fit_gp(x2,y2)
# mu = GP2.mean(xtest)
# plt.plot(xtest,mu, 'r-', label = 'huber-true-model-corupted')

GP = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d, loss = 'huber', huber_delta=1.3)
GP.fit_gp(x,y)
mu = GP.mean(xtest)
plt.plot(xtest,mu, '--', color = "tab:orange", label = 'huber-uncorrupted', lw = 3)
plt.legend()

plt.plot(x,y, 'ro', ms = 5)

plt.plot(xnew,ynew, 'ko', ms = 10)
plt.show()
# GP.fit_gp(x,y2)
# GP.optimize_params(type="bandwidth", restarts=10, verbose = False, optimizer = 'pytorch-minimize', scale = 10.)
# GP.visualize(xtest, show = True, fig = False, color = 'yellow')
#

