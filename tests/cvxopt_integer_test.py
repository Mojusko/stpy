import numpy as np
import cvxopt
import torch
from cvxopt import glpk,solvers
from stpy.continuous_processes.gauss_procc import GaussianProcess
import matplotlib.pyplot as plt
N = 128
d = 30

# Rotation
theta = np.radians(45.)
thetainv = np.pi - theta
c, s = np.cos(theta), np.sin(theta)
Q = torch.from_numpy(np.array(((c, -s), (s, c))))
M = torch.randn(size = (d,d), dtype = torch.float64)
[Q,R] = torch.qr(M)


def solve(Q,c,n = 10, verbose = True, up = None, low = None):
	print ("Starting Acq. Fucn solver...")
	print ("Resolution: ", n)

	# Grid

	tau = torch.from_numpy(np.arange(-n,n+1,1).astype(np.double))/n
	s = torch.ones(2*n+1)
	Tau = torch.zeros(size = (d,d*(2*n+1)), dtype = torch.float64)
	S = torch.zeros(size = (d,d*(2*n+1)), dtype = torch.float64)
	for j in range(d):
		Tau[j,j*(2*n+1):(j+1)*(2*n+1)] = tau
		S[j, j * (2 * n + 1):(j + 1) * (2 * n + 1)] = s

	B = Q @ Tau

	if (up is not None) or (low is not None):
		G = torch.cat((B, -B, S, -S, torch.t(c),-torch.t(c)))
		h = torch.ones(4 * d + 2)
		h[0:2 * d] = 1
		h[3 * d:4 * d] = -1
		h[4 * d ] = up
		h[4 * d + 1] = -low
	else:
		G = torch.cat((B, -B, S, -S))
		h = torch.ones(4 * d)
		h[0:2 * d] = 1
		h[3 * d:4 * d] = -1
	# Indicator variables
	x = torch.zeros(size = (d*(2*n+1),1),dtype = torch.float64)
	print (h)
	cc = cvxopt.matrix(c.view(-1).numpy().astype(np.double))
	Gc = cvxopt.matrix(G.numpy().astype(np.double))
	hc = cvxopt.matrix(h.numpy().astype(np.double))

	glpk.options['it_lim'] = 10
	
	solvers.solve(solver=cp.CBC)
	(status, x)= glpk.ilp(cc,Gc,hc,B=set(range(d*(2*n+1)))  )

	return x

# def N is the desired resolution
tau = torch.from_numpy(np.arange(-N,N+1,1).astype(np.double))/N
gp = GaussianProcess(gamma = 0.5, s = 0.001)
c = torch.randn(size = (d*(2*N+1),1), dtype = torch.float64)
for i in range(d):
	plt.plot(gp.sample(tau.view(-1,1)).numpy())
	c[i*(2*N+1):(i+1)*(2*N+1)] = gp.sample(tau.view(-1,1))
plt.show()

def select(c,N,n, low, up):
	plt.subplot(211)
	cs = torch.randn(size = (d*(2*n+1),1), dtype = torch.float64)
	step = N//n
	plt.plot(c.numpy())
	for i in range(d):
		for j in range(2*n+1):
			cs[i*(2*n+1)+j] = c[i*(2*N+1)+(j*step)]
			plt.plot(i*(2*N+1)+(j*step),cs[i*(2*n+1)+j].numpy(),"ro")

	sum_c = c[0*(2*N+1):(0+1)*(2*N+1)] *0
	for i in range(d):
		sum_c = sum_c+ c[i*(2*N+1):(i+1)*(2*N+1)]
	if low is not None:
		plt.subplot(2, 1, 2)
		plt.plot(sum_c.numpy())
		plt.plot(sum_c.numpy()*0+low,"--", label = "low")
		plt.plot(sum_c.numpy() * 0 + up, "--", label = "up")
	plt.legend()

	plt.show()
	return cs


up = None
low = None
L = 0.01

x = solve(Q, c, n=N, up=up, low=low)

for j in range(int(np.log2(N))):
	n = np.power(2,j)

	print(N, n)
	cs = select(c,N,n, low,up )
	x = solve(Q,cs,n = n, up=up, low = low)
	up = float(torch.mm(torch.t(cs),torch.from_numpy(np.array(x))))
	low = float(torch.mm(torch.t(cs),torch.from_numpy(np.array(x)))) - L/n

"""
m_value = 0
for i in range(d):
	qq = c[i*(2*n+1):(i+1)*(2*n+1)]
	m_value +=torch.min(torch.sort(qq)[0])
print (m_value)

# solve simple LP with additional constraints of binarity
I = torch.eye(  d*(2*n+1)  ,dtype = torch.float64)
G = torch.cat((G,I,-I))
h = torch.ones(4*d+2*d*(2*n+1))
h[3*d:4*d] = -1
h[4*d+d*(2*n+1):] = 0

#print (h)
cc = cvxopt.matrix(c.view(-1).numpy().astype(np.double))
Gc = cvxopt.matrix(G.numpy().astype(np.double))
hc = cvxopt.matrix(h.numpy().astype(np.double))

solvers.options['abstol']=10e-10
solvers.options['reltol']=10e-10
#(status, xlp)= glpk.ilp(cc,Gc,hc)

#res = solvers.lp(cc,Gc,hc)
#print (res['x'])
#print (x)

"""