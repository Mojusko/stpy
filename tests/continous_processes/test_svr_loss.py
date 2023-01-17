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
s = 0.1
B = 0.001
x = torch.rand(N,d).double()*2 - 1
xtest = torch.from_numpy(interval(n,d,L_infinity_ball=1))

# true
GP_true = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d)
ytest = GP_true.sample(xtest)
GP_true.fit_gp(xtest,ytest)



y = GP_true.mean(x).clone()
xnew = x[0,:].view(1,1) + eps
ynew = torch.rand(size = (1,1))*B
y2 = torch.vstack([y,ynew])
x2 = torch.vstack([x,xnew])

GP  = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d, loss = 'svr', svr_eps=0.1, s = s)
GP2 = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d, s = s, loss = "squared")
GP3 = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d, s = s, loss = 'unif', B = B)
GP4 = GaussianProcess(gamma=0.1, kernel_name="squared_exponential", d=d, s = s, loss = 'huber', huber_delta=1.35)

GP.fit_gp(x,y)
GP2.fit_gp(x,y)
GP3.fit_gp(x,y)
GP4.fit_gp(x,y)

plt.plot(x,y, 'ro', label = 'points')
plt.plot(xtest, ytest, 'b-', label = "truth")

plt.plot(xtest, GP.mean_std(xtest)[0], 'g-', label = "svr")
plt.plot(xtest, GP2.mean_std(xtest)[0], 'r-', label = "squared")
plt.plot(xtest, GP3.mean_std(xtest)[0], 'y-', label = "unif")

plt.plot(xtest, GP4.mean_std(xtest)[0], '--', color = "orange", label = "huber")
plt.legend()
plt.show()


GP.fit_gp(x2,y2)
GP2.fit_gp(x2,y2)
GP3.fit_gp(x2,y2)
GP4.fit_gp(x2,y2)

plt.plot(x,y, 'ro', label = 'points')
plt.plot(xnew,ynew, 'ko', label = 'corrupted')
plt.plot(xtest, ytest, 'b-', label = "truth")

plt.plot(xtest, GP.mean_std(xtest)[0], 'g-', label = "svr")
plt.plot(xtest, GP2.mean_std(xtest)[0], 'r-', label = "squared")
plt.plot(xtest, GP3.mean_std(xtest)[0], 'y-', label = "unif")
plt.plot(xtest, GP4.mean_std(xtest)[0], '--', color = "orange", label = "huber")

plt.legend()
plt.show()




GP.fit_gp(x2,y2)
GP2.fit_gp(x2,y2)
GP3.fit_gp(x2,y2)
GP4.fit_gp(x2,y2)

plt.plot(x,y, 'ro', label = 'points')
plt.plot(xnew,ynew, 'ko', label = 'corrupted')
plt.plot(xtest, ytest, 'b-', label = "truth")

GP.fit_gp(x2,y2)
GP.optimize_params(type="bandwidth", restarts=5, verbose = False, optimizer = 'pytorch-minimize', scale = 1.)
GP2.fit_gp(x2,y2)
GP2.optimize_params(type="bandwidth", restarts=5, verbose = False, optimizer = 'pytorch-minimize', scale = 1.)
GP3.fit_gp(x2,y2)
#GP3.optimize_params(type="bandwidth", restarts=5, verbose = False, optimizer = 'pytorch-minimize', scale = 1.)
GP4.fit_gp(x2,y2)
GP4.optimize_params(type="bandwidth", restarts=5, verbose = False, optimizer = 'pytorch-minimize', scale = 1.)

plt.plot(xtest, GP.mean_std(xtest)[0], 'g-', label = "svr")
plt.plot(xtest, GP2.mean_std(xtest)[0], 'r-', label = "squared")
plt.plot(xtest, GP3.mean_std(xtest)[0], 'y-', label = "unif")
plt.plot(xtest, GP4.mean_std(xtest)[0], '--', color = "orange", label = "huber")



plt.legend()
plt.show()
