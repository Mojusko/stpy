import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.kernels import KernelFunction
from stpy.helpers.helper import interval
import matplotlib.pyplot as plt
import numpy as np

N = 100
n = 40
d = 2
eps = 0.01
s = 1
x = torch.rand(N,d).double()*2 - 1
xtest = torch.from_numpy(interval(n,d,L_infinity_ball=1))

# true
GP = GaussianProcess(kernel_name="ard_matern", d=d)
y = GP.sample(x)
GP.fit_gp(x,y)
GP.optimize_params(type="bandwidth", restarts=5, verbose = False, optimizer = 'pytorch-minimize', scale = 1., weight=1.)
GP.visualize_contour(xtest)
#

