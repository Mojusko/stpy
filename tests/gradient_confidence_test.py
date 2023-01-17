import numpy as np
import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.helpers.helper import interval
#%matplotlib notebook


# 2D Grid
n = 20
n_vis = 50
d = 2
xtest_vis = torch.from_numpy(interval(n_vis,d))
xtest = torch.from_numpy(interval(n,d,L_infinity_ball=1.))
noise_s = 0.001
bw = 0.4

GP_true = GaussianProcess(groups = [[0],[1]], gamma = bw*torch.ones(2,dtype = torch.float64), kernel = "ard", s = noise_s)
y = GP_true.sample(xtest)
GP_true.fit_gp(xtest,y)

zero = torch.from_numpy(np.array([[0.,0.]]))
gradient, hessian = GP_true.mean_gradient_hessian(zero, hessian = True)


GP_fit = GaussianProcess(gamma = bw, kernel = "squared_exponential", s = noise_s)
GP_fit.fit_gp(xtest ,y)
#GP_fit.visualize(xtest_vis)
GP_fit.log_marginal_likelihood_self()

GP_fit.visualize_quiver(xtest_vis)


print ("Zero:" ,zero)
g, V = GP_fit.gradient_mean_var(zero)

print (gradient)

print (V)

print ("------------------")