import matplotlib.pyplot as plt
from stpy.continuous_processes.fourier_fea import GaussianProcessFF
from stpy.embeddings.polynomial_embedding import PolynomialEmbedding
from stpy.test_functions.benchmarks import *
from doexpy.bandits.OPPR_TS_GP import OPPR_TS_GP

N = 1
s = 0.00001
n = 50
L_infinity_ball = 0.5
d = 2

thetae = np.radians(35.)
ce, se = np.cos(thetae), np.sin(thetae)
R = torch.from_numpy(np.array(((ce, -se), (se, ce))))
D = torch.diag(torch.Tensor([0.8, 1.1]).double())
#D = torch.diag(torch.Tensor([1, 1]).double())

W = R.T @ D @ R
print (W)
BenchmarkFunc = QuadraticBenchmark(d=d, R=W)

x = BenchmarkFunc.initial_guess(N)
xtest = BenchmarkFunc.interval(n)
gamma = BenchmarkFunc.bandwidth()
bounds = BenchmarkFunc.bounds()
BenchmarkFunc.scale_max(xtest=xtest)


F = lambda x: BenchmarkFunc.eval(x, sigma=0)
F0 = lambda x: BenchmarkFunc.eval(x, sigma=0)


def plot_contour(xtest,ytest,lim=None):
    from scipy.interpolate import griddata
    xx = xtest[:, 0].numpy()
    yy = xtest[:, 1].numpy()
    grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
    grid_z_mu = griddata((xx, yy), ytest[:, 0].numpy(), (grid_x, grid_y), method='linear')
    fig, ax = plt.subplots(figsize=(10, 9))
    cs = ax.contourf(grid_x, grid_y, grid_z_mu)
    ax.contour(cs, colors='k')
    if lim is not None:
        plt.xlim([-lim,lim])
        plt.ylim([-lim,lim])
    plt.colorbar(cs)
    # Plot grid.
    ax.grid(c='k', ls='-', alpha=0.1)



## Additive Model
m = 64
GP = GaussianProcessFF(d=d, s=s, m = torch.ones(d)*m, gamma=gamma*torch.ones(d), bounds=bounds, groups = stpy.helpers.helper.full_group(d))

## Global Model
# m = 512
# gamma = 0.05
# embedding = HermiteEmbedding(gamma=gamma, m=m, d=d, diameter=1, approx = "hermite")
# Map = lambda x: embedding.embed(x)

p = 5
d = 2
embedding = PolynomialEmbedding(d,p)
Map = lambda x: embedding.embed(x)

# Starting points
x0_1 = torch.Tensor([0.1, 0.1]).double().view(-1, d)

#x0_1 = torch.Tensor([-0.1, 0.]).double().view(-1, d)
x0_2 = torch.Tensor([0.1, 0.1]).double().view(-1, d)

print("Embeding size:", Map(x0_1).size())


Bandit = OPPR_TS_GP(x0_1, F, GP, Map, finite_dim=True, s = s, GPMap = True)
#Bandit.decolerate(x0_1,10e-5,1)
Bandit.decolerate_AJD([x0_1,x0_2],10e-5,1)

print (Bandit.Q)
print (W@Bandit.Q)
print (W@torch.inverse(Bandit.Q))

