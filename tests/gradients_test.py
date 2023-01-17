import numpy as np
import matplotlib.pyplot as plt
import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.helpers.helper import interval
#%matplotlib notebook


# 2D Grid
for n in np.arange(50,60,10):
    n_vis = 50
    d = 2
    xtest_vis = torch.from_numpy(interval(n_vis,d))
    xtest = torch.from_numpy(interval(n,d,L_infinity_ball=0.01))
    noise_s = 0.001
    bw = 0.4

    GP_true = GaussianProcess(groups = [[0],[1]], gamma = bw*torch.ones(2,dtype = torch.float64), kernel = "ard", s = noise_s)
    y = GP_true.sample(xtest)
    GP_true.fit_gp(xtest,y)

    zero = torch.from_numpy(np.array([[0.,0.]]))
    gradient, hessian = GP_true.mean_gradient_hessian(zero, hessian = True)

   # print ("gradient:",gradient)
   # print ("hessian:",hessian)


    # [mu, _] = GP_true.get_lambdas(2, mean=True)
    # for z in [10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6, 10e-7]:
    #     print(z, stpy.helper.finite_differences(mu,z,xtest[0].view(1,-1)))

    theta = np.radians(12.)
    thetainv = np.pi - theta
    c, s = np.cos(theta), np.sin(theta)
    RandRot = torch.from_numpy(np.array(((c,-s), (s, c))))
    #print (RandRot)

    def eval(x):
        xprime = x.mm(RandRot)
        f = GP_true.mean_std(xprime)[0]
        return f


    y_prime = eval(xtest)
    GP_fit = GaussianProcess(groups = [[0,1]], gamma = bw*torch.ones(2,dtype = torch.float64), kernel = "ard", s = noise_s)
    GP_fit.fit_gp(xtest,y_prime)
    GP_fit.visualize(xtest_vis)
    GP_fit.log_marginal_likelihood_self()

    print ("Zero:",zero)
    g, V = GP_fit.gradient_mean_var(zero)

    print (gradient)

    print (V)

    print ("------------------")

    gradient, hessian = GP_fit.mean_gradient_hessian(zero, hessian = True)
    Q = torch.symeig(hessian, eigenvectors = True)[1]


    print(GP_fit.mean_std(zero))
    #print ("Estimated:",Q)
    #print ("True:", RandRot)
    P = torch.t(Q) @ RandRot
    I = torch.eye(GP_fit.d, dtype = torch.float64)
    Noise = s*I*s
    Perm = torch.clamp(torch.abs(P), min=10e-3)
    print (n, P,torch.norm(torch.abs(P)-Perm))


    no = 100
    thetas = np.linspace(0.,np.pi,no)
    res = []
    for theta in thetas:
        c, s = np.cos(theta), np.sin(theta)
        Rot = np.array(((c,-s), (s, c)))
        Rot = torch.from_numpy(Rot)
        res.append(float(GP_fit.log_marginal_likelihood(GP_fit.kernel_object.gamma,Rot,GP_fit.kernel_object.kappa)))
    plt.plot(thetas,res)
    plt.plot([thetainv],np.average(np.array(res)),'ro')
    plt.show()

    GP_fit.optimize_params(type = "rots", restarts = 10)
    GP_fit.log_marginal_likelihood_self()

    print(GP_fit.Rot)
    P = GP_fit.Rot @ RandRot
