import torch
import stpy

if __name__ == "__main__":

	### Generate data - a sample from a Gaussian process
	n = 1024
	N = 5
	gamma = 0.09
	#gamma = 1.
	s = 0.2
	# benchmark = stpy.test_functions.benchmarks.GaussianProcessSample(d =1, gamma = gamma, sigma = s, n = n)
	benchmark = stpy.test_functions.benchmarks.Simple1DFunction(d=1, sigma=s)
	for j in range(10):
		m = (2*(j+1)) ** 2
		#m = 64
		x = benchmark.initial_guess(N, adv_inv=False)
		y = benchmark.eval(x)
		xtest = benchmark.interval(1024)

		#print (x)
		CFF = stpy.continuous_processes.fourier_fea.GaussianProcessFF(gamma=gamma, approx="ccff", m=m, s=s)
		QFF = stpy.continuous_processes.fourier_fea.GaussianProcessFF(gamma=gamma, approx="hermite", m=m, s=s)
		TFF = stpy.continuous_processes.fourier_fea.GaussianProcessFF(gamma=gamma, approx="trapezoidal", m=m, s=s)

		K1 = TFF.embed(x)@TFF.embed(x).T
		K2 = QFF.embed(x) @ QFF.embed(x).T
		K3 = CFF.embed(x) @ CFF.embed(x).T
		#	print(K2)
		# print("----------------")
		#print(K3)
		# print("----------------")
		print(m, torch.norm(K1 - K2), torch.norm(K2 -K3))

	#CFF.fit_gp(x,y)
	#CFF.visualize(xtest)
