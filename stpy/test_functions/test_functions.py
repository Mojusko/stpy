from h5py import File
import stpy
import stpy.continuous_processes.gauss_procc
from stpy.test_functions.neural_net import train_network
from tensorflow.examples.tutorials.mnist import input_data
from stpy.helpers.helper import *
import pickle




def isin(element, test_elements,  assume_unique=False):
	(n,d) = element.shape
	(m,d) = test_elements.shape
	maskFull = np.full((n), False, dtype = bool)
	for j in range(m):
		mask = np.full((n), True, dtype=bool)
		for i in range(d):
			#mask = np.logical_and(mask,np.in1d(element[:,i],test_elements[j,i], assume_unique=assume_unique))
			mask = np.logical_and(mask, np.isclose(element[:, i], test_elements[j, i], atol=1e-01))
			#print (j, i, mask)
		maskFull = np.logical_or(mask,maskFull)
	#print (maskFull)
	return maskFull

class test_function: 

	def __init__(self):
		"nothing"
		self.sampled = False
		self.init = False
		self.scale = 1.0

	## General F
	def f(self,X,sigma = 0.00001,a = 0.5):
		# in X rows are points, cols are features
		X = X*8
		y = -np.sin(a*np.sum(X**2,axis = 1)).reshape(X.shape[0],1)
		y = y + sigma*np.random.randn(X.shape[0],1)
		return y

	def f_bounds(self,N,n,d = 1, L_infinity_ball = 1.):
		x = np.random.uniform(-L_infinity_ball, L_infinity_ball, size=(N,d))
		#grid
		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(-L_infinity_ball,L_infinity_ball,n).reshape(n,1) for i in range(d)]
			xtest = cartesian(arrays)
		return (d,xtest,x,0.15)

	def f_opt_bounds(self, d = 1, L_infinity_ball = 1):
		b = tuple([(-L_infinity_ball,L_infinity_ball) for i in range(d)])
		return b


	def optimize_f(self, d = 1,  a = 0.5, L_infinity_ball = 1):
		from scipy.optimize import minimize

		grad = lambda x: np.cos(np.sum(x**2)/2)*x
		fun  = lambda x: np.sin(np.sum(x**2)/2)+1

		bounds = self.f_opt_bounds(d = d, L_infinity_ball = L_infinity_ball)
		r = []
		for _ in range(500):
			(d,_,x,_) = self.f_bounds (1,None, d = d, L_infinity_ball = L_infinity_ball)
			x0 = x[0,:]
			res = minimize(fun, x0, method = "SLSQP", jac = grad, tol = 0.0001, bounds=bounds)
			r.append(fun(res.x))
		
		print (d, max(r))












	def sample_ss(self,X, sigma = 0.001, gamma = 1.0, GP = None):
		# in X rows are points, cols are features
		if self.sampled == False:
			#print ("sampling")
			if GP == None:
				GP = stpy.continuous_processes.gauss_procc.GaussianProcess(s = sigma, gamma = gamma)
				self.sample = GP.sample(torch.from_numpy(self.xtest)).numpy()
				mask = isin(self.xtest,X)
				self.sampled = True
				return self.sample[mask,:].numpy() + np.random.randn(X.shape[0],1)*sigma
			else:
				self.sample = GP.sample(torch.from_numpy(self.xtest)).numpy()
				mask = isin(self.xtest,X)
				self.sampled = True
				return self.sample[mask,:] + np.random.randn(X.shape[0],1)*sigma
		else:
			mask = isin(self.xtest,X)
			return self.sample[mask,:] + np.random.randn(X.shape[0],1)*sigma

	def sample_ss_bounds(self,N,n,d = 1, L_infinity_ball = 1., gamma = 1.0):
		#self.sampled = False
		#grid
		arrays = [np.linspace(-L_infinity_ball,L_infinity_ball,n).reshape(n,1) for i in range(d)]
		xtest = cartesian(arrays)
		self.xtest = xtest
		self.n = n 
		#x = self.xtest[np.random.randint(0,n,size = N),:]
		x = self.xtest[np.random.permutation(np.arange(0,self.xtest.shape[0],1))[0:N],:]
		x = np.sort(x,axis = 0)
		return (d,xtest,x,gamma)
	
	def sample_ss_reset(self):
		self.samples = False



	def optimize(self,xtest,ytest,groups,s):
		(n,d) = xtest.size()
		kernel = stpy.kernels.KernelFunction(kernel_name="ard", gamma=torch.ones(d, dtype=torch.float64) * 0.1, groups=groups)
		GP = stpy.continuous_processes.gauss_procc.GaussianProcess(kernel_custom=kernel, s=s, d=d)
		GP.fit_gp(xtest,ytest)
		GP.optimize_params( type="bandwidth" )
		print ("Optimized")
		return torch.min(kernel.gamma)


	## Branin Function 
	def branin(self,X,sigma = 0.1):
		if X.shape[1] != 2:
			raise AssertionError("Invalid dimension of grid with Branin Function")
		else:
			xx = X[:,0]
			yy = X[:,1]
			y = ((yy - (5.1/(4.*np.pi))*(xx**2) + 5./np.pi - 6.)**2 + 10.*(1.- 1./(8.*np.pi))*np.cos(xx) + 10.)/150
			y = -y.reshape(X.shape[0],1)
			return y

	def branin_bounds(self,N,n):
		x = np.random.uniform(0, 10, size=(N,2))
		#grid
		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(-5,10,n).reshape(n,1), np.linspace(0,15,n).reshape(n,1)]
			xtest = cartesian(arrays)
		return (2,xtest,x,2.5)

	def branin_opt_bounds(self):
		b = tuple([(-5,10),(0,15)])
		return b 


	## Camelback Function 
	def camelback(self,X,sigma = 0.1):
		if X.shape[1] != 2:
			raise AssertionError("Invalid dimension of grid with Branin Function")
		else:
			xx = X[:,0]*4
			yy = X[:,1]*2
			y = (4. - 2.1*xx**2 + (xx**4)/3.)*(xx**2) + xx*yy + (-4. + 4*(yy**2))*(yy**2)
			y = -y.reshape(X.shape[0],1)
			#y = np.tanh(y)
			y = y/5.
			return y/self.scale + sigma*np.random.randn(X.shape[0],1)

	def camelback_bounds(self,N,n, adv_inv = False):
		if adv_inv == False:
			x = np.random.uniform(-0.5, 0.5	, size=(N,2))
		else:
			x = np.random.uniform(-0.5, -0.4, size=(N, 2))
		#grid
		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(-0.5,0.5,n).reshape(n,1), np.linspace(-0.5,0.5,n).reshape(n,1)]
			xtest = cartesian(arrays)
		return (2,xtest,x,0.1)

	def camelback_opt_bounds(self):
		b = tuple([(-0.5,0.5),(-0.5,0.5)])
		return b 

	def camelback_scale(self,xtest):
		self.scale = np.max((self.camelback(xtest,sigma = 0)))
		print ("Scaling:", self.scale)



	## Hartmann 6 
	def hartmann6(self,X,sigma = 0.1):
		if X.shape[1] != 6:
			raise AssertionError("Invalid dimension of grid with Branin Function")
		else:
			#opt = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
			#fopt = np.array([[-3.32237]])
	 
			alpha = [1.00, 1.20, 3.00, 3.20]
			A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
							   [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
							   [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
							   [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
			P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
										[2329, 4135, 8307, 3736, 1004, 9991],
										[2348, 1451, 3522, 2883, 3047, 6650],
										[4047, 8828, 8732, 5743, 1091, 381]])
	 
			"""6d Hartmann test function
				input bounds:  0 <= xi <= 1, i = 1..6
				global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
				min function value = -3.32237
			"""
	 
			external_sum = 0
			for i in range(4):
				internal_sum = 0
				for j in range(6):
					internal_sum = internal_sum + A[i, j] * (X[:, j] - P[i, j]) ** 2
				external_sum = external_sum + alpha[i] * np.exp(-internal_sum)
	 
			return external_sum[:, np.newaxis]

	def hartmann6_bounds(self,N,n):
		x = np.random.uniform(0, 1, size=(N,6))
		#grid
		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(0,1,n).reshape(n,1) for i in range(6)]
			xtest = cartesian(arrays)
		return (6,xtest,x,0.5)

	def hartmann6_opt_bounds(self):
		b = tuple([(0,1) for i in range(6)])
		return b


	## Hartmann 4
	def hartmann4(self,X,sigma = 0.1):
		if X.shape[1] != 4:
			raise AssertionError("Invalid dimension of grid with Branin Function")
		else:
	 
			alpha = [1.00, 1.20, 3.00, 3.20]

			A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
							   [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
							   [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
							   [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
			
			P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
										[2329, 4135, 8307, 3736, 1004, 9991],
										[2348, 1451, 3522, 2883, 3047, 6650],
										[4047, 8828, 8732, 5743, 1091, 381]])
	 
			"""6d Hartmann test function
				input bounds:  0 <= xi <= 1, i = 1..6
				global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
				min function value = -3.32237
			"""
	 
			external_sum = 0
			for i in range(4):
				internal_sum = 0
				for j in range(4):
					internal_sum = internal_sum + A[i, j] * (X[:, j] - P[i, j]) ** 2
				external_sum = external_sum + alpha[i] * np.exp(-internal_sum)
	 
			return external_sum[:, np.newaxis]

	def hartmann4_bounds(self,N,n):
		x = np.random.uniform(0, 1, size=(N,4))
		#grid
		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(0,1,n).reshape(n,1) for i in range(4)]
			xtest = cartesian(arrays)
		return (4,xtest,x,0.5)

	def hartmann4_opt_bounds(self):
		b = tuple([(0,1) for i in range(4)])
		return b






	def hartmann3(self,X,sigma = 0.1):
	 
		X_lower = np.array([0, 0, 0])
		X_upper = np.array([1, 1, 1])
		#opt = np.array([[0.114614, 0.555649, 0.852547]])
		#fopt = np.array([[-3.86278]])
		alpha = [1.0, 1.2, 3.0, 3.2]
		A = np.array([[3.0, 10.0, 30.0],
							   [0.1, 10.0, 35.0],
							   [3.0, 10.0, 30.0],
							   [0.1, 10.0, 35.0]])
		P = 0.0001 * np.array([[3689, 1170, 2673],
										[4699, 4387, 7470],
										[1090, 8732, 5547],
										[381, 5743, 8828]])
	 
		external_sum = 0
		for i in range(4):
			internal_sum = 0 
			for j in range(3):
				internal_sum = internal_sum + A[i, j] * (X[:, j] - P[i, j]) ** 2

			external_sum = external_sum + alpha[i] * np.exp(-internal_sum)

		return external_sum[:, np.newaxis]

	def hartmann3_bounds(self,N,n):
		x = np.random.uniform(0, 1, size=(N,3))
		#grid
		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(0,1,n).reshape(n,1) for i in range(3)]
			xtest = cartesian(arrays) 

		return (3,xtest,x,0.4)


	def hartmann3_opt_bounds(self):
		b = tuple([(0,1) for i in range(3)])
		return b


	def michal_old(self, X, sigma = 0.1):
		(n,d) = X.shape
		sum_ = np.zeros(shape = (X.shape[0],1))

		for ii in range(d):
			xi = X[:,ii]
			#print ("xi",xi)
			i = ii+1
			new = np.sin(xi) * np.power((np.sin(i * np.power(xi, 2) / np.pi)), (2*d))
			sum_ += new.reshape(n,1)
		return -0.5*sum_ + np.random.randn(X.shape[0],1)*sigma

	def stang_old(self, X, sigma=0.1):
		(n, d) = X.shape
		sum_ = np.zeros(shape=(X.shape[0], 1))

		for ii in range(d):
			xi = X[:, ii]
			new = xi ** 4 - 16. * xi ** 2 + 5 * xi
			sum_ += new.reshape(n, 1)

		sum_ = sum_ / (38.7122 * d)
		# sum_ = sum_/d

		return -0.5 * sum_ + np.random.randn(X.shape[0], 1) * sigma



	def michal_un(self, X, sigma = 0.1):
		(n,d) = X.shape
		X = (X+ 0.5)*np.pi
		ar = np.arange(1,d+1,1)
		sum_ = np.sin(X) * np.power((np.sin(ar * X / np.pi)), (2*d))
		sum_ = np.sum(sum_,axis = 1).reshape(-1,1)
		return sum_ + np.random.randn(X.shape[0],1)*sigma


	def michal(self, X, sigma = 0.1):
		(n,d) = X.shape
		X = (X + 0.5)*np.pi
		ar = np.arange(1,d+1,1)
		sum_ = np.sin(X) * np.power((np.sin(ar * X / np.pi)), (2*d))
		sum_ = np.sum(sum_,axis = 1).reshape(-1,1)
		sum_ = sum_/self.michal_optimum(d)[1]
		return sum_ + np.random.randn(X.shape[0],1)*sigma

	def michal_bounds(self,N,n, d = 1, adv_inv = False):
		if adv_inv == False:
			x = np.random.uniform(-0.5, 0.5	, size=(N,d))
		else:
			x = np.random.uniform(-0.5, 0., size=(N, d))

		if n == None:
			xtest = None
		else: 
			arrays = [np.linspace(-0.5,0.5,n).reshape(n,1) for i in range(d)]
			xtest = cartesian(arrays)

		return (d,xtest,x,0.3)

	def michal_opt_bounds(self,d):
		b = tuple([(-0.5,0.5) for i in range(d)])
		return b

	def	michal_optimum(self,d):
		q = 20
		opt = np.ones(shape	=(q))
		opt[0] = 2.93254
		opt[1] = 2.34661
		opt[2] =1.64107
		opt[3] =1.24415
		opt[4] =0.999643
		opt[5] = 0.834879
		opt[6] = 2.1089
		opt[7] = 1.84835
		opt[8] = 1.64448
		opt[9] = 1.48089
		opt[10] = 1.34678
		opt[11] =1.2349
		opt[12] = 1.89701
		opt[13] = 1.76194
		opt[14] = 1.64477
		opt[15] = 1.54218
		opt[16] = 1.45162
		opt[17] = 1.37109
		opt[18] = 1.81774
		opt = opt[0:d].reshape(1,-1)
		opt = (opt/np.pi)-0.5
		value = self.michal_un(opt,sigma = 0)
		return (opt,value[0][0])



	def stang_un(self, X, sigma = 0.1):
		(n,d) = X.shape
		X = X*8
		Y = X**2
		sum_ = np.sum(Y**2 - 16.*Y + 5*X, axis = 1).reshape(-1,1)
		sum_ = sum_
		return -0.5*sum_ + np.random.randn(X.shape[0],1)*sigma

	def stang(self, X, sigma = 0.1):
		(n,d) = X.shape
		X = X*8
		Y = X**2
		sum_ = np.sum(Y**2 - 16.*Y + 5*X, axis = 1).reshape(-1,1)
		sum_ = sum_/self.stang_optimum(d)[1]
		return -0.5*sum_ + np.random.randn(X.shape[0],1)*sigma


	def stang_bounds(self,N,n, d = 1, adv_inv = False):
		if adv_inv == False:
			x = np.random.uniform(-0.5, 0.5	, size=(N,d))
		else:
			print ("Adversarially initiallized")
			x = np.random.uniform(0.4, 0.5, size=(N, d))

		if n == None:
			xtest = None
		else: 
			arrays = [np.linspace(-0.5,0.5,n).reshape(n,1) for i in range(d)]
			xtest = cartesian(arrays)

		return (d,xtest,x,0.6)

	def stang_opt_bounds(self,d):
		b = tuple([(-0.5,0.5) for i in range(d)])
		return b

	def stang_optimum(self,d):
		opt = np.ones(shape = (d))*(-2.9035)
		opt = opt/8
		opt = opt.reshape(1,-1)

		value = self.stang_un(opt,sigma = 0.0)
		return (opt,value[0][0])








	def double_group_un(self, X, sigma = 0.1):
		sum_ = np.sum(np.exp(-(np.diff(X,axis = 1)/0.25)**2),axis =1 ).reshape(-1,1)
		return 0.5*sum_ + np.random.randn(X.shape[0],1)*sigma

	def double_group(self, X, sigma = 0.1):
		(n,d) = X.shape
		sum_ = np.sum(np.exp(-(np.diff(X,axis = 1)/0.25)**2),axis =1 ).reshape(-1,1)
		sum_ = sum_/self.double_group_optimum(d)[1]
		return 0.5*sum_ + np.random.randn(X.shape[0],1)*sigma

	def double_group_bounds(self,N,n, d = 1, adv_inv = False):
		if adv_inv == False:
			x = np.random.uniform(-0.5, 0.5	, size=(N,d))
		else:
			print ("Adversarially initiallized")
			x = np.random.uniform(-0.5, -0.4, size=(N, d))

		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(-0.5,0.5,n).reshape(n,1) for i in range(d)]
			xtest = cartesian(arrays)

		return (d,xtest,x,0.6)

	def double_group_opt_bounds(self,d):
		b = tuple([(-0.5,0.5) for i in range(d)])
		return b

	def double_group_optimum(self,d):
		opt = np.zeros(shape = (1,d))
		value = self.double_group_un(opt,0)[0][0]
		return (opt,value)






	def swissfel(self,X,sigma = 0.1):
		if self.init == False:
			raise AssertionError("Need to run bounds first.")
		else:
			if sigma == 0.0:
				return self.model.predict(X)[0]
			else:
				return self.model.predict(X)[0] + np.random.randn(X.shape[0], 1) * self.noise

	def swissfel_bounds(self,N,n):
		if self.init == False:
			import os.path
			fname = "/home/mojko/Documents/PhD/RFFinBO/code/test_problems/swissfel_model.p"
			if not os.path.isfile(fname):
				f = File('/home/mojko/Documents/PhD/RFFinBO/code/test_problems/evaluations.hdf5')
				dset = f['1']
				X = dset["x"][:].reshape(-1, 5)

				# y response and scale
				Y = dset["y"][:].reshape(-1, 1)
				Y = Y / np.max(np.abs(Y))

				# noise structure
				Yerr = dset["y_std"]/np.max(np.abs(Y))
				self.noise = np.std(Yerr)
				print ("Estimated noise level", self.noise)

				# data scale to [-0.5,0.5]
				X = dset["x"][:].reshape(-1, 5)
				for j in range(5):
					a = np.min(X[:,j])
					b = np.max(X[:,j])
					X[:,j] = (X[:,j]/(b-a)) -0.5 - a/(b-a)

				## fully additive kernel s
				self.kernel = GPy.kern.RBF(1,active_dims=[0]) + GPy.kern.RBF(1,active_dims=[1])  \
							  + GPy.kern.RBF(1,active_dims=[2])  + GPy.kern.RBF(1,active_dims=[3])  \
							  + GPy.kern.RBF(1,active_dims =[4])
				self.model = GPy.models.GPRegression(X, Y, self.kernel)
				print ("Model fit")
				self.model.optimize(messages=True)
				print ("ML likelihood fit")
				self.init = True
				# save pickle
				pickle.dump(self.model, open("/home/mojko/Documents/PhD/RFFinBO/code/test_problems/swissfel_model.p", "wb"))
				pickle.dump(self.noise, open("/home/mojko/Documents/PhD/RFFinBO/code/test_problems/swissfel_noise.p", "wb"))
			else:
				self.init = True
				self.model = pickle.load(open("/home/mojko/Documents/PhD/RFFinBO/code/test_problems/swissfel_model.p", "rb"))
				self.noise = pickle.load(open("/home/mojko/Documents/PhD/RFFinBO/code/test_problems/swissfel_noise.p", "rb"))

		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(-0.5,0.5,n).reshape(n,1) for i in range(5)]
			xtest = cartesian(arrays)

		#bw = np.min(self.kernel.lengthscale)
		x = np.random.uniform(-0.5,0.5,size=(N,5))
		return (5,xtest,x,0.1)

	def swissfel_opt_bounds(self):
		b = tuple([(-0.5, 0.5) for i in range(5)])
		return b

	def swissfel_optimum(self):
		from scipy.optimize import minimize
		#maximize the function
		mybounds = self.swissfel_opt_bounds()
		fun = lambda x: -self.swissfel(x.reshape(1, -1), sigma=0.0)[0][0]

		best = -10.
		repeats = 10
		for i in range(repeats):
			x0 = np.random.uniform(-0.5,0.5, size = (5,))
			res = minimize(fun, x0, method="L-BFGS-B", tol=0.0001, bounds=mybounds)
			value = self.swissfel(res.x.reshape(1,-1), sigma=0)
			if value > best:
				best = value
				self.opt_loc = res.x.reshape(1,-1)
		return (self.opt_loc,best)





	def neural_net(self, X, sigma = 0.1):
		(n,d) = X.shape
		res = []
		val_size = 400
		if self.sampled == False:
			self.sampled = True
			try:
				self.mnist = input_data.read_data_sets("~/.", one_hot=True, validation_size = val_size)
			except:
				self.mnist = input_data.read_data_sets("~/.", one_hot=True)

		for x in X:
			(it, acc) = train_network(self.mnist, dropout = x[0], verbose = False, 
				val_size = val_size, maxiter = 300 , initialization_params = x[1:], no_filters_1=self.NN, 
				no_filters_2=self.NN2, val_count = 30)
			res.append(acc)

		return np.array(acc).reshape(n,1)


	def neural_net_bounds(self,N,n, NN = 16, NN2 = 22):
		self.NN = NN
		self.NN2 = NN2
		d = self.NN+self.NN2

		x = np.random.uniform(0, 10, size=(N,d))
		dropout = np.random.uniform(0,1, size= (N,1) )
		x = np.concatenate((x,dropout), axis =1)
		
		if n == None:
			xtest = None
		else:
			arrays = [np.linspace(0,1,n).reshape(n,1)] + [np.linspace(0,10,n).reshape(n,1) for i in range(d)]
			xtest = cartesian(arrays)
		
		return (d+1,xtest,x,0.9)

	def neural_net_opt_bounds(self):
		d = self.NN + self.NN2
		b = tuple([(0,1)] + [(0,10) for i in range(d)])
		return b


if __name__=="__main__":
	s = 0
	TT = test_function()
	Fs = [lambda x: TT.f(x,sigma = s),lambda x: TT.branin(x,sigma = s),lambda x: TT.camelback(x,sigma = s),lambda x: TT.hartmann3(x,sigma = s),lambda x: TT.hartmann4(x,sigma = s),lambda x: TT.hartmann6(x,sigma = s)]
	Fbounds = [lambda n: TT.f_bounds(1,n), lambda n: TT.branin_bounds(1,n),lambda n: TT.camelback_bounds(1,n),lambda n: TT.hartmann3_bounds(1,n),lambda n: TT.hartmann4_bounds(1,n),lambda n: TT.hartmann6_bounds(1,n)]
	ns = [4000,200,200,100,50,10]
	tests = ["1D","Branin","Camelback","Hartmann3","Hartmann4","Hartmann6"]
	z = []
	for i in range(6):
		(d, xtest, x, _) = Fbounds[i](ns[i])
		z.append(np.max(Fs[i](xtest)))
		print(tests[i],np.max(Fs[i](xtest)))
	print (z)

	for d,n in zip([1,2,3,4],[900,100,50,3]):
		G = lambda x: TT.stang(x, sigma = s)
		(q,xtest,x,_) = TT.stang_bounds(1, n, d = d)
		print (d, np.max(G(xtest)), np.max(G(xtest))/d)

	# G = lambda x: TT.michal(x, sigma = s)
	# (d,xtest,x,_) = TT.michal_bounds(1,5, d = 10)
	# print (d, np.max(G(xtest)), np.max(G(xtest))/d)

	# for d in np.arange(1,31,1):
	# 	TT.optimize_f(d = d)


	print ("==== Optimized vs Non-Optimized ==== ")
	print ("Michal")
	multistart = 400
	d = 10
	G1 = lambda x: TT.michal(x, sigma = 0.)
	fun = lambda x: -TT.michal(x.reshape(-1,1), sigma = 0.)[0][0]
	(d,xtest,x,_) = TT.michal_bounds(20,None, d = d)
	mybounds = TT.michal_opt_bounds(d = d)

	from scipy.optimize import minimize

	results = []
	for i in range(multistart):
		x0 = np.random.randn(d)
		for i in range(d):
			x0[i] = np.random.uniform(mybounds[i][0],mybounds[i][1])
		res = minimize(fun, x0, method = "L-BFGS-B", jac = None, tol = 0.00001, bounds=mybounds)
		#res = minimize(fun, x0, method = "SLSQP", jac = None, tol = 0.00001, bounds=mybounds)
		solution = res.x
		results.append([solution,-fun(solution)])
	results = np.array(results)
	print (np.max(results[:,1]))
	
	print ("Stybtang")
	for d in [10,20]:
		multistart = 400
		G1 = lambda x: TT.stang(x, sigma = 0.)
		fun = lambda x: -TT.stang(x.reshape(-1,1), sigma = 0.)[0][0]
		(d,xtest,x,_) = TT.stang_bounds(20,None, d = d)
		mybounds = TT.stang_opt_bounds(d = d)
		from scipy.optimize import minimize

		results = []
		for i in range(multistart):
			x0 = np.random.randn(d)
			for i in range(d):
				x0[i] = np.random.uniform(mybounds[i][0],mybounds[i][1])
			res = minimize(fun, x0, method = "L-BFGS-B", jac = None, tol = 0.00001, bounds=mybounds)
			#res = minimize(fun, x0, method = "SLSQP", jac = None, tol = 0.00001, bounds=mybounds)
			solution = res.x
			results.append([solution,-fun(solution)])

		results = np.array(results)
		print (d, np.max(results[:,1]))

		#print (G1(x))
	#print (G2(x))





