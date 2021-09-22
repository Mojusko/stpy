import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from stpy.helpers.coreset_helper import coreset_leverage_score_greedy
from stpy.helpers.helper import cartesian
from stpy.embeddings.positive_embedding import PositiveEmbedding
from scipy.interpolate import BPoly,PPoly
from stpy.borel_set import BorelSet
from stpy.continuous_processes.nystrom_fea import NystromFeatures

class BernsteinEmbedding(PositiveEmbedding):

	def __init__(self, *args, **kwargs):
		super().__init__(*args,**kwargs)

	def basis_fun(self, x, j): #1d
		"""
		Return the value of basis function \phi_j(x)
		:param x: double, need to be in the interval
		:param j: integer, index of hat functions, 0 <= j <= m-1
		:return: \phi_j(x)
		"""
		lim = [self.interval[0],self.interval[1]]
		c = np.zeros(shape = (self.m,1))
		c[j] = 1
		bp = BPoly(c, lim, extrapolate=False)
		res = bp(x.numpy())
		value = torch.from_numpy(np.nan_to_num(res))
		return value
		#return torch.from_numpy(bp(x.numpy()))


	def get_polynomial(self,j):
		if self.d == 1:
			lim = [self.interval[0], self.interval[1]]
			c = np.zeros(shape=(self.m, 1))
			c[j] = 1
			roots = PPoly.from_bernstein_basis(BPoly(c, lim)).roots()
			poly = np.polynomial.polynomial.Polynomial(np.polynomial.polynomial.polyfromroots(roots), domain = np.array(lim))

		elif self.d == 2:
			lim = [self.interval[0], self.interval[1]]
			k = j // self.m
			l = j % self.m
			c = np.zeros(shape=(self.m, 1))
			c[k] = 1
			bp = BPoly(c, lim)
			c = np.zeros(shape=(self.m, 1))
			c[l] = 1
			bp2 = BPoly(c, lim)
			roots1 = PPoly.from_bernstein_basis(bp).roots()
			roots2 =PPoly.from_bernstein_basis(bp2).roots()
			poly1 = np.polynomial.polynomial.Polynomial( np.polynomial.polynomial.polyfromroots(roots1), domain = np.array(lim))
			poly2 = np.polynomial.polynomial.Polynomial( np.polynomial.polynomial.polyfromroots(roots2), domain = np.array(lim))
			poly = poly1*poly2
		return poly

	def integral(self, S):
		assert( S.d == self.d)
		psi = torch.zeros(self.get_m()).double()

		if self.d == 1:
			a, b = float(S.bounds[0, 0]), float(S.bounds[0, 1])
			for j in range(self.get_m()):
				lim = [self.interval[0], self.interval[1]]
				c = np.zeros(shape=(self.m, 1))
				c[j] = 1
				bp = BPoly(c, lim)
				xa = np.maximum(self.interval[0],a)
				xb = np.minimum(self.interval[1] ,b)
				psi[j] = bp.integrate(xa,xb, extrapolate=False)

		elif self.d == 2:
			xa,xb = S.bounds[0,0],S.bounds[0,1]
			ya,yb = S.bounds[1,0],S.bounds[1,1]
			for j in range(self.get_m()):
				lim = [self.interval[0], self.interval[1]]

				k = j //self.m
				l = j % self.m

				c = np.zeros(shape=(self.m, 1))
				c[k] = 1
				bp = BPoly(c, lim)
				vol1 = bp.integrate(xa, xb)
				c = np.zeros(shape=(self.m, 1))
				c[l] = 1
				bp = BPoly(c, lim)
				vol2 = bp.integrate(ya, yb)
				psi[j] = vol1*vol2

		Gamma_half = self.cov()
		return psi @ Gamma_half

	def product_integral(self,S):
		m = self.get_m()
		Psi = torch.zeros(size = (m,m)).double()
		a,b = S.bounds[0,0],S.bounds[0,1]
		for i in range(m):
			for j in range(m):
				P = self.get_polynomial(i)*self.get_polynomial(j)
				new_p = P.integ()
				xb = np.minimum(new_p.domain[1],b)
				xa = np.maximum(new_p.domain[0], a)
				Psi[i,j] = new_p(xb) - new_p(xa)
				print (i,j,Psi[i,j])
		Gamma_half = self.cov()
		return Gamma_half @ Psi @ Gamma_half.T


	# def cov(self, inverse = False):
	# 	if self.precomp == False:
	# 		dm = (self.interval[1] - self.interval[0]) / (self.m - 1)  # delta m
	# 		t = self.interval[0] + torch.linspace(0, self.m - 1, self.m) * dm
	# 		t = self.borel_set.return_legendre_discretization(self.m)[0]
	# 		if self.d == 1:
	# 			t = t.view(-1, 1).double()
	# 		Z = self.embed_internal(t)
	# 		Gamma = self.kernel(t, t)
	# 		self.Gamma_half = torch.cholesky(Gamma + self.s * self.s * torch.eye(Gamma.size()[0]).double())
	# 		self.Gamma_half = self.Gamma_half
	# 		self.precomp = True
	# 		Z = self.embed(t)
	# 	else:
	# 		pass
	# 	return self.Gamma_half

class BernsteinSplinesOverlapping(PositiveEmbedding):

	def __init__(self, *args,degree = 4, **kwargs):
		super().__init__(*args,**kwargs)
		self.degree = degree

	def basis_fun(self, x, q, derivative=0, extrapolate=False):  # 1d
		"""
		Return the value of basis function \phi_j(x)

		:param x: double, need to be in the interval
		:param j: integer, index of hat functions, 0 <= j <= m-1
		:return: \phi_j(x)
		"""

		j = q// (self.degree//2)
		k = q % (self.degree//2)

		dm = (self.interval[1] - self.interval[0]) / ( (self.m//(self.degree//2)) )  # delta m
		tj = self.interval[0] + j * dm
		lim = [tj, tj + 2*dm]

		c = np.zeros(shape = (self.degree//2,1))
		c[k] = 1.
		bp = BPoly(c, lim)
		res = bp(x.numpy(), nu=derivative, extrapolate=extrapolate)

		if extrapolate == False:
			mask = x.numpy()== (tj+dm/2)
			res[mask] = np.nan
		value = torch.from_numpy(np.nan_to_num(res))
		return value

	def integral(self, S):
		assert( S.d == self.d)
		psi = torch.zeros(self.get_m()).double()

		if self.d == 1:
			a, b = float(S.bounds[0, 0]), float(S.bounds[0, 1])
			for q in range(self.get_m()):

				j = q // self.degree
				k = q % self.degree

				dm = (self.interval[1] - self.interval[0]) / ((self.m // self.degree))  # delta m
				tj = self.interval[0] + j * dm
				lim = [tj, tj + dm]
				c = np.zeros(shape=(self.degree, 1))
				c[k] = 1.
				bp = BPoly(c, lim)
				xa = np.maximum(tj,a)
				xb = np.minimum(tj+dm,b)
				psi[q] = np.nan_to_num(bp.integrate(xa,xb,extrapolate= False))

		elif self.d ==2:
			xa, xb = S.bounds[0, 0], S.bounds[0, 1]
			ya, yb = S.bounds[1, 0], S.bounds[1, 1]
			for z in range(self.get_m()):
				q1 = z // self.m
				q2 = z % self.m

				j1 = q1 // self.degree
				k1 = q1 % self.degree
				j2 = q2 // self.degree
				k2 = q2 % self.degree


				dm = (self.interval[1] - self.interval[0]) / ((self.m // self.degree))  # delta m
				tj1 = self.interval[0] + j1 * dm
				tj2 = self.interval[0] + j2 * dm
				lim1 = [tj1, tj1 + dm]
				lim2 = [tj2, tj2 + dm]
				c = np.zeros(shape=(self.degree, 1))
				c[k1] = 1.
				bp = BPoly(c, lim1)
				vol1 = bp.integrate(xa,xb)
				c = np.zeros(shape=(self.degree, 1))
				c[k2] = 1.
				bp = BPoly(c, lim2)
				vol2 = bp.integrate(ya, yb)
				psi[z] = vol1*vol2

		Gamma_half = self.cov()
		return psi @ Gamma_half



class BernsteinSplinesEmbedding(PositiveEmbedding):

	def __init__(self, *args,degree = 4, derivatives = 2, **kwargs):
		super().__init__(*args,**kwargs)
		self.degree = degree
		self.derivatives = derivatives

	#def basis_fun(self, x, j, k, derivative = 0, extrapolate = False): #1d
	def basis_fun(self, x, q, derivative=0, extrapolate=False):  # 1d
		"""
		Return the value of basis function \phi_j(x)

		:param x: double, need to be in the interval
		:param j: integer, index of hat functions, 0 <= j <= m-1
		:return: \phi_j(x)
		"""

		j = q//self.degree
		k = q % self.degree

		dm = (self.interval[1] - self.interval[0]) / ( (self.m//self.degree) )  # delta m
		tj = self.interval[0] + j * dm


		lim = [tj, tj + dm]
		c = np.zeros(shape = (self.degree,1))
		c[k] = 1.
		bp = BPoly(c, lim)
		res = bp(x.numpy(), nu=derivative, extrapolate=extrapolate)

		if extrapolate == False:
			mask = x.numpy()== (tj+dm)
			res[mask] = np.nan
		value = torch.from_numpy(np.nan_to_num(res))
		return value

	def embed_internal_derivative(self, x, l=1, extrapolate = False):
		if self.d == 1:
			out = torch.zeros(size = (x.size()[0],self.m), dtype = torch.float64)
			for j in range(0,self.m,1):
				out[:, j] = self.basis_fun(x, j, derivative=l, extrapolate=extrapolate).view(-1)
			return out

	def get_constraints(self):
		s = self.m**self.d

		# positivity constraints
		l = np.full(s, self.b)
		u = np.full(s, self.B)
		I = np.identity(s)


		# pointwise fix
		Zs = []
		vs = []
		for j in range(self.derivatives+1):
			no_nodes = (self.m//self.degree)-1
			Z = np.zeros(shape=(no_nodes,s))
			dm = (self.interval[1] - self.interval[0]) / ((self.m//self.degree) )  # delta m

			for i in range(no_nodes):
				ti = torch.from_numpy(np.array(self.interval[0] + (i+1) * dm)).view(1,-1)
				Z[i,i*self.degree:i*self.degree+self.degree] = self.embed_internal_derivative(ti, l = j, extrapolate = True).numpy().reshape(-1)[i*self.degree:i*self.degree+self.degree]
				Z[i,(i+1)*self.degree:(i+1)*self.degree+self.degree] = -self.embed_internal_derivative(ti, l = j, extrapolate = True).numpy().reshape(-1)[(i+1)*self.degree:(i+1)*self.degree+self.degree]
			v = np.zeros(self.m//self.degree-1)
			Zs.append(Z)
			vs.append(v)

		Lambda = np.concatenate([I] + Zs)
		l = np.concatenate( [l] + vs)
		u = np.concatenate([u] + vs)
		return (l,Lambda,u)



	def integral(self, S):
		assert( S.d == self.d)
		psi = torch.zeros(self.get_m()).double()

		if self.d == 1:
			a, b = float(S.bounds[0, 0]), float(S.bounds[0, 1])
			for q in range(self.get_m()):

				j = q // self.degree
				k = q % self.degree

				dm = (self.interval[1] - self.interval[0]) / ((self.m // self.degree))  # delta m
				tj = self.interval[0] + j * dm
				lim = [tj, tj + dm]
				c = np.zeros(shape=(self.degree, 1))
				c[k] = 1.
				bp = BPoly(c, lim)
				xa = np.maximum(tj,a)
				xb = np.minimum(tj+dm,b)
				psi[q] = np.nan_to_num(bp.integrate(xa,xb,extrapolate= False))

		elif self.d ==2:
			xa, xb = S.bounds[0, 0], S.bounds[0, 1]
			ya, yb = S.bounds[1, 0], S.bounds[1, 1]
			for z in range(self.get_m()):
				q1 = z // self.m
				q2 = z % self.m

				j1 = q1 // self.degree
				k1 = q1 % self.degree
				j2 = q2 // self.degree
				k2 = q2 % self.degree


				dm = (self.interval[1] - self.interval[0]) / ((self.m // self.degree))  # delta m
				tj1 = self.interval[0] + j1 * dm
				tj2 = self.interval[0] + j2 * dm
				lim1 = [tj1, tj1 + dm]
				lim2 = [tj2, tj2 + dm]
				c = np.zeros(shape=(self.degree, 1))
				c[k1] = 1.
				bp = BPoly(c, lim1)
				vol1 = bp.integrate(xa,xb)
				c = np.zeros(shape=(self.degree, 1))
				c[k2] = 1.
				bp = BPoly(c, lim2)
				vol2 = bp.integrate(ya, yb)
				psi[z] = vol1*vol2

		Gamma_half = self.cov()
		return psi @ Gamma_half



	def product_integral(self,S):
		pass
if __name__ == "__main__":
	from stpy.continuous_processes.gauss_procc import GaussianProcess
	from stpy.helpers.helper import interval
	import matplotlib.pyplot as plt
	from stpy.continuous_processes.kernelized_features import KernelizedFeatures
	from stpy.embeddings.embedding import HermiteEmbedding
	from stpy.kernels import KernelFunction
	from stpy.embeddings.bump_bases import FaberSchauderEmbedding
	d = 1
	m = 32
	n = 64
	N = 10

	sqrtbeta = 2
	s = 0.001
	b = 0.0
	B = 200

	gamma = 0.1
	kernel_object = KernelFunction(gamma = gamma)

	#Emb = BernsteinSplinesEmbedding(d, m,kernel_object=kernel_object, offset=0.5,b=b,B=B,s = s)
	EmbBern = BernsteinEmbedding(d,m,kernel_object=kernel_object,offset=0.5,b=b,B=B,s = s)
	EmbFaber = FaberSchauderEmbedding(d,m,kernel_object=kernel_object,offset=0.5,b=b,B=B,s = s)
	GP = GaussianProcess(d = d, s = s, kernel=kernel_object)
	#GPNyst = KernelizedFeatures(embedding=EmbNys.GP,m = m, s = s,)

	xtest = torch.from_numpy(interval(n,d,L_infinity_ball=1.1))
	x = torch.from_numpy(np.random.uniform(-1,1,N)).view(-1,1)

	F_true = lambda x: torch.sin(x)**2-0.1
	F = lambda x: F_true(x) + s*torch.randn(x.size()[0]).view(-1,1).double()
	y = F(x)



	#Emb.fit_gp(x,y)
	EmbBern.fit_gp(x,y)
	EmbFaber.fit_gp(x,y)

	GP.fit_gp(x,y)

	#mu = Emb.mean_std(xtest)
	mu_true,_ = GP.mean_std(xtest)
	mu_bern = EmbBern.mean_std(xtest)
	mu_faber = EmbFaber.mean_std(xtest)

	plt.plot(xtest, xtest*0+b, 'k--')
	#plt.plot(xtest, xtest * 0 + B, 'k--')

	plt.plot(xtest,F_true(xtest),'r', label = 'true')
	#plt.plot(xtest,mu_true_nyst,color = 'lightblue', label = 'Nystrom')
	plt.plot(xtest,mu_true,'b--', label = 'no-constraints')

	plt.plot(x,y,'ro')
	#plt.plot(xtest, mu, 'g-x', label = 'splines Bernstein')
	plt.plot(xtest, mu_bern, 'y-o',label = 'Bernstein basis')
	plt.plot(xtest, mu_faber, 'g-o', label='Faber basis')
	plt.legend()
	plt.show()