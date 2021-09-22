import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class RandomProcess:


	def visualize_function(self,xtest,f_true, filename = None):
		from mpl_toolkits.mplot3d import axes3d, Axes3D
		d = xtest.size()[1]
		if d == 1:
			plt.plot(xtest,f_true(xtest))
		elif d == 2:
			from scipy.interpolate import griddata
			#plt.figure(figsize=(15, 7))
			#plt.clf()
			ax = plt.axes(projection='3d')
			xx = xtest[:, 0].numpy()
			yy = xtest[:, 1].numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z = griddata((xx, yy), f_true(xtest)[:, 0].numpy(), (grid_x, grid_y), method='linear')
			ax.plot_surface(grid_x, grid_y, grid_z, color='b', alpha=0.4)
			if filename is not None:
				plt.xticks(fontsize=20, rotation=0)
				plt.yticks(fontsize=20, rotation=0)
				plt.savefig(filename, dpi = 300)
			plt.show()




	def visualize_function_contous(self,xtest,f_true, filename = None, levels = 10, figsize = (15,7)):
		from mpl_toolkits.mplot3d import axes3d, Axes3D
		d = xtest.size()[1]
		if d ==1:
			pass
		elif d == 2:
			from scipy.interpolate import griddata
			xx = xtest[:, 0].numpy()
			yy = xtest[:, 1].numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			f = f_true(xtest)
			grid_z_f = griddata((xx, yy), f[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			fig, ax = plt.subplots(figsize=figsize)
			cs = ax.contourf(grid_x, grid_y, grid_z_f,levels= levels)
			ax.contour(cs, colors='k')
			cbar = fig.colorbar(cs)
			#if self.x is not None:
			#	ax.scatter(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), c='r', s=100, marker="o")
			ax.grid(c='k', ls='-', alpha=0.1)

			if filename is not None:
				plt.xticks(fontsize=24, rotation=0)
				plt.yticks(fontsize=24, rotation=0)
				plt.savefig(filename, dpi = 300)
			#plt.show()

	def visualize(self,xtest,f_true = None, points = True, show = True, size = 2,
				  norm = 1, fig = True, sqrtbeta = 2, constrained = None, d = None, matheron_kernel=None):
		from mpl_toolkits.mplot3d import axes3d, Axes3D

		[mu, std] = self.mean_std(xtest)

		if d is None:
			d = self.d

		if d == 1:
			if fig == True:
				plt.figure(figsize=(15, 7))
				plt.clf()
			if self.x is not None:
				plt.plot(self.x.detach().numpy(), self.y.detach().numpy(), 'r+', ms=10, marker="o")
			if size > 0:

				if matheron_kernel is not None:
					z = self.sample_matheron(xtest,matheron_kernel, size=size).numpy().T
				else:
					z  = self.sample(xtest, size=size).numpy().T

				for z_arr,label in zip(z,['sample']+[None for _ in range(size-1)]):
					plt.plot(xtest.view(-1).numpy(),z_arr, 'k--', lw = 2, label = label)

			plt.fill_between(xtest.numpy().flat, (mu - sqrtbeta * std).numpy().flat, (mu + sqrtbeta * std).numpy().flat,color="#dddddd")
			if f_true is not None:
				plt.plot(xtest.numpy(),f_true(xtest).numpy(),'b-',lw = 2, label = "truth")
			plt.plot(xtest.numpy(), mu.numpy(), 'r-', lw=2, label="posterior mean")
			#plt.title('Posterior mean prediction plus 2 st.deviation')
			plt.legend()
			if show == True:
				plt.show()

		elif d == 2:
			from scipy.interpolate import griddata
			plt.figure(figsize=(15,7))
			plt.clf()
			ax = plt.axes(projection='3d')
			xx = xtest[:, 0].numpy()
			yy = xtest[:, 1].numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			if f_true is not None:
				grid_z = griddata((xx, yy), f_true(xtest)[:,0].numpy(), (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z, color='b', alpha=0.4, label = "truth")
			if points == True and self.fit == True:
				ax.scatter(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), self.y[:,0].detach().numpy(), c='r', s=100, marker="o", depthshade=False)
			if self.beta is not None:
				beta = self.beta(norm = norm)
				grid_z2 = griddata((xx, yy), (mu.detach()+beta*std.detach())[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z2, color='gray', alpha=0.2)
				grid_z3 = griddata((xx, yy), (mu.detach()-beta*std.detach())[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z3, color='gray', alpha=0.2)

			ax.plot_surface(grid_x, grid_y, grid_z_mu, color='r', alpha=0.4)
			#plt.title('Posterior mean prediction plus 2 st.deviation')
			plt.show()

		else:
			print("Visualization not implemented")

	def visualize_subopt(self,xtest,f_true = None, points = True, show = True, size = 2, norm = 1, fig = True, beta = 2):
		from mpl_toolkits.mplot3d import axes3d, Axes3D
		[mu, std] = self.mean_std(xtest)

		print ("Visualizing in: ", self.d, "dimensions...")

		if self.d == 1:
			if fig == True:
				plt.figure(figsize=(15, 7))
				plt.clf()
			if self.x is not None:
				plt.plot(self.x.detach().numpy(), self.y.detach().numpy(), 'r+', ms=10, marker="o")
			plt.plot(xtest.numpy(), self.sample(xtest, size=size).numpy(), 'k--', lw=2, label="sample")
			plt.fill_between(xtest.numpy().flat, (mu - 2 * std).numpy().flat, (mu + 2 * std).numpy().flat,color="#dddddd")
			if f_true is not None:
				plt.plot(xtest.numpy(),f_true(xtest).numpy(),'b-',lw = 2, label = "truth")
			plt.plot(xtest.numpy(), mu.numpy(), 'r-', lw=2, label="posterior mean")

			min = torch.max(mu - beta*std)
			mask = (mu + beta*std < min)
			v = torch.min(mu - beta * std).numpy()-1
			plt.plot(xtest.numpy()[mask], 0*xtest.numpy()[mask]+v,'ko', lw = 6,label = "Discarted Region")



			plt.title('Posterior mean prediction plus 2 st.deviation')
			plt.legend()

			if show == True:
				plt.show()

	def visualize_slice(self,xtest,slice, show = True, eps = None, size = 1, beta = 2):
		append = torch.ones(size = (xtest.size()[0],1), dtype=torch.float64)*slice
		xtest2 = torch.cat((xtest,append), dim = 1)

		[mu, std] = self.mean_std(xtest2)

		plt.figure(figsize=(15, 7))
		plt.clf()
		plt.plot(xtest.numpy(), self.sample(xtest, size=size).numpy(), 'k--', lw=2, label="sample")
		print(std.size(), mu.size())
		if self.x is not None:
			plt.plot(self.x[:,0].detach().numpy(), self.y.detach().numpy(), 'r+', ms=10, marker="o")
		plt.fill_between(xtest.numpy().flat, (mu - 2 * std).numpy().flat, (mu + 2 * std).numpy().flat, color="#dddddd")
		plt.fill_between(xtest.numpy().flat, (mu + 2 * std).numpy().flat, (mu + 2 * std + 2*self.s).numpy().flat, color="#bbdefb")
		plt.fill_between(xtest.numpy().flat, (mu - 2 * std - 2*self.s).numpy().flat, (mu - 2 * std).numpy().flat, color="#bbdefb")

		if eps is not None:
			mask = (beta*std < eps)
			v = torch.min(mu - beta * std - 2*self.s).numpy()
			plt.plot(xtest.numpy()[mask], 0*xtest.numpy()[mask]+v,'k', lw = 6,label = "$\\mathcal{D}_E$ - $\\epsilon$ accurate domain in a subspace")

		plt.plot(xtest.numpy(), mu.numpy(), 'r-', lw=2, label="posterior mean")
		plt.title('Posterior mean prediction plus 2 st.deviation')
		plt.legend()
		if show == True:
			plt.show()



	def visualize_contour_with_gap(self,xtest,f_true = None, gap = None, show = False):
		[mu, _] = self.mean_std(xtest)

		if self.d == 2:
			from scipy.interpolate import griddata
			xx = xtest[:, 0].detach().numpy()
			yy = xtest[:, 1].detach().numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')

			fig, ax = plt.subplots(figsize=(15, 7))
			cs = ax.contourf(grid_x, grid_y, grid_z_mu)
			ax.contour(cs, colors='k')

			ax.plot(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), 'ro', ms=10)
			cbar = fig.colorbar(cs)

			ax.grid(c='k', ls='-', alpha=0.1)

			if f_true is not None:
				f = f_true(xtest)
				grid_z_f = griddata((xx, yy), f[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
				fig, ax = plt.subplots(figsize=(15, 7))
				cs = ax.contourf(grid_x, grid_y, grid_z_f)
				ax.contour(cs, colors='k')
				cbar = fig.colorbar(cs)
				ax.grid(c='k', ls='-', alpha=0.1)
			if show == True:
				plt.show()

	def visualize_contour(self,xtest,f_true = None, show = True, points = True, ms = 5, levels = 20):
		[mu, _] = self.mean_std(xtest)

		if self.d == 2:
			from scipy.interpolate import griddata
			xx = xtest[:, 0].detach().numpy()
			yy = xtest[:, 1].detach().numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			fig, ax = plt.subplots(figsize=(15, 7))
			cs = ax.contourf(grid_x, grid_y, grid_z_mu)
			ax.contour(cs, colors='k')
			if points == True:
				ax.plot(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), 'wo', ms=ms, alpha = 0.5)
			cbar = fig.colorbar(cs)
			ax.grid(c='k', ls='-', alpha=0.1)

			if f_true is not None:
				f = f_true(xtest)
				grid_z_f = griddata((xx, yy), f[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
				fig, ax = plt.subplots(figsize=(15, 7))
				cs = ax.contourf(grid_x, grid_y, grid_z_f, levels = levels)
				ax.contour(cs, colors='k')
				cbar = fig.colorbar(cs)
				ax.grid(c='k', ls='-', alpha=0.1)
			if show == True:
				plt.show()
			return ax

	def visualize_quiver(self,xtest, size = 2,norm = 1):
		from mpl_toolkits.mplot3d import axes3d, Axes3D
		[mu, std] = self.mean_std(xtest)
		if self.d == 2:
			from scipy.interpolate import griddata
			plt.figure(figsize=(15,7))
			plt.clf()
			ax = plt.axes(projection='3d')
			xx = xtest[:, 0].detach().numpy()
			yy = xtest[:, 1].detach().numpy()
			grid_x, grid_y = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):100j]
			grid_z_mu = griddata((xx, yy), mu[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			#

			ax.scatter(self.x[:, 0].detach().numpy(), self.x[:, 1].detach().numpy(), self.y[:,0].detach().numpy(), c='r', s=100, marker="o", depthshade=False)

			if self.beta is not None:
				beta = self.beta(norm = norm)
				grid_z2 = griddata((xx, yy), (mu.detach()+beta*std.detach())[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z2, color='gray', alpha=0.2)
				grid_z3 = griddata((xx, yy), (mu.detach()-beta*std.detach())[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
				ax.plot_surface(grid_x, grid_y, grid_z3, color='gray', alpha=0.2)

			ax.plot_surface(grid_x, grid_y, grid_z_mu, color='r', alpha=0.4)
			plt.title('Posterior mean prediction plus 2 st.deviation')


			derivatives = torch.zeros(xtest.size()[0],2)
			for index,point in enumerate(xtest):
				derivatives[index,:] = self.mean_gradient_hessian(point.view(-1,2))
				print (derivatives[index,:] )

			print (derivatives.size())


			grid_der_x_mu = griddata((xx, yy), derivatives[:, 0].detach().numpy(), (grid_x, grid_y), method='linear')
			grid_der_y_mu = griddata((xx, yy), derivatives[:, 1].detach().numpy(), (grid_x, grid_y), method='linear')

			fig, ax = plt.subplots(figsize=(15, 7))
			cs = ax.contourf(grid_x, grid_y, grid_z_mu)

			ax.contour(cs, colors='k')

			# Plot grid.
			ax.grid(c='k', ls='-', alpha=0.1)
			ax.quiver(grid_x, grid_y, grid_der_x_mu, grid_der_y_mu)

			plt.show()

		else:
			print("Visualization not implemented")


if __name__ == "__main__":
	from stpy.continuous_processes.gauss_procc import GaussianProcess
	from stpy.continuous_processes.fourier_fea import GaussianProcessFF
	from stpy.continuous_processes.kernelized_features import KernelizedFeatures
	from stpy.kernels import KernelFunction
	from stpy.embeddings.embedding import HermiteEmbedding, RFFEmbedding
	import stpy
	import torch
	import matplotlib.pyplot as plt
	import numpy as np

	n = 1024
	N = 256
	gamma = 0.09
	s = 0.1
	# benchmark = stpy.test_functions.benchmarks.GaussianProcessSample(d =1, gamma = gamma, sigma = s, n = n)
	benchmark = stpy.test_functions.benchmarks.Simple1DFunction(d=1, sigma=s)

	x = benchmark.initial_guess(N, adv_inv=True)
	y = benchmark.eval(x)
	xtest = benchmark.interval(1024)

	# GP = GaussianProcess(gamma=gamma, s=s)
	# GP.fit_gp(x, y)
	# GP.visualize(xtest, show=False, size=5)
	# plt.show()

	m = 64
	kernel = KernelFunction(gamma=gamma)
	embedding = HermiteEmbedding(gamma=gamma, m=m)
	RFF = KernelizedFeatures(embedding=embedding, s=s, m=m)
	RFF.fit_gp(x, y)
	RFF.visualize(xtest, fig = False, show=False, size=5, matheron_kernel = kernel)
	plt.show()