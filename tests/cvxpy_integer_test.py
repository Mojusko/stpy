import cvxpy as cp
import numpy as np
import torch
from stpy.continuous_processes.gauss_procc import GaussianProcess
import matplotlib.pyplot as plt

N = 32
d = 20

# Rotation
theta = np.radians(45.)
thetainv = np.pi - theta
c, s = np.cos(theta), np.sin(theta)
Q = torch.from_numpy(np.array(((c, -s), (s, c))))
M = torch.randn(size=(d, d), dtype=torch.float64)
[Q, R] = torch.qr(M)


def solve(Q, c, n=10, verbose=True, up=None, low=None, xwarm = None):
	if verbose == True:
		print("Starting Acq. Fucn solver...")
		print("Resolution: ", n)
	# Grid

	tau = torch.from_numpy(np.arange(-n, n + 1, 1).astype(np.double)) / n
	s = torch.ones(2 * n + 1)
	Tau = torch.zeros(size=(d, d * (2 * n + 1)), dtype=torch.float64)
	S = torch.zeros(size=(d, d * (2 * n + 1)), dtype=torch.float64)
	for j in range(d):
		Tau[j, j * (2 * n + 1):(j + 1) * (2 * n + 1)] = tau
		S[j, j * (2 * n + 1):(j + 1) * (2 * n + 1)] = s

	B = Q @ Tau

	if (up is not None) or (low is not None):
		G = torch.cat((B, -B, S, -S, torch.t(c), -torch.t(c)))
		h = torch.ones(4 * d + 2)
		h[0:2 * d] = 1
		h[3 * d:4 * d] = -1
		h[4 * d] = up
		h[4 * d + 1] = -low
	else:
		G = torch.cat((B, -B, S, -S))
		h = torch.ones(4 * d)
		h[0:2 * d] = 1
		h[3 * d:4 * d] = -1
	# Indicator variables

	x = cp.Variable(d * (2 * n + 1), boolean=True)
	if xwarm is not None:
		x.value = xwarm.numpy()
	c = c.view(-1).numpy()

	objective = cp.Maximize(c * x)
	constraints = [0 <= x, x <= 1, G.numpy()*x <= h.view(-1).numpy()]
	prob = cp.Problem(objective, constraints)
	prob.solve(solver=cp.MOSEK,verbose=verbose, warm_start=True)


	return (x.value,Tau.numpy() @ x.value, np.dot(c,x.value))


# def N is the desired resolution
tau = torch.from_numpy(np.arange(-N, N + 1, 1).astype(np.double)) / N
gp = GaussianProcess(gamma=0.5, s=0.001)
c = torch.randn(size=(d * (2 * N + 1), 1), dtype=torch.float64)
for i in range(d):
	z = gp.sample(tau.view(-1, 1))
	plt.plot(z.numpy())
	c[i * (2 * N + 1):(i + 1) * (2 * N + 1)] = z
plt.show()


def select(c, N, n, val):
	cs = torch.randn(size=(d * (2 * n + 1), 1), dtype=torch.float64)
	if val is not None:
		sol = torch.randn(size=(d * (2 * n + 1), 1), dtype=torch.float64).view(-1)*0
	else:
		sol = None
	step = N // n

	for i in range(d):
		#plt.plot(c[i * (2 * n + 1):(i+1) * (2 * n + 1)].numpy())
		for j in range(2 * n + 1):
			cs[i * (2 * n + 1) + j] = c[i * (2 * N + 1) + (j * step)]
			if val is not None:
				if (c[i * (2 * N + 1) + (j * step)] - val[i])**2 < 10e-10:
					sol[i * (2 * N + 1) + (j * step)] = 1.0
			#plt.plot((i * (2 * N + 1) + (j * step))/((i+1)*N), cs[i * (2 * n + 1) + j].numpy(), "ro")
	#plt.show()
	return cs,sol


up = None
low = None
L = 10e20

#x = solve(Q, c, n=N, up=up, low=low)
sol = None
val = None

for j in range(int(np.log2(N))+1):
	n = np.power(2, j)

	print(N, n)
	cs, sol = select(c, N, n, val)
	x , val = solve(Q, cs, n=n, up=up, low=low, xwarm = sol)
	print (x, val)
	#up = float( torch.dot(cs.view(-1),torch.from_numpy(x)))
	#low = float( torch.dot(cs.view(-1),torch.from_numpy(x))) - L/n
	sol = x

plt.figure()
colors = ['b','k','r','g','y']
for i in range(d):
	z = c[i * (2 * N + 1):(i + 1) * (2 * N + 1)].view(-1).numpy()
	x = np.linspace(-1,1,2*N+1)
	plt.plot(x,z, color = colors[i % 5], label = str(i))
	index = np.argmin(z)
	plt.plot(val[i],z[index],'o', color = colors[i % 5],label = str(i), ms = 10)
#plt.legend()
plt.show()