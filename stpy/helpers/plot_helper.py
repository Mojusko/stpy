
def plot_ellipse(offset, cov, scale=1, theta_num=1e3, axis=None, plot_kwargs=None, fill=False, fill_kwargs=None):
	'''
	offset = 2d array which gives center of ellipse
	cov = covariance of ellipse
	scale = scale ellipse by constant factor
	theta_num = used for a linspace below, not sure exactly (?)

	'''
	# Get Ellipse Properties from cov matrix

	eig_vec, eig_val, u = np.linalg.svd(cov)
	# Make sure 0th eigenvector has positive x-coordinate
	if eig_vec[0][0] < 0:
		eig_vec[0] *= -1

	semimaj = np.sqrt(eig_val[0])
	semimin = np.sqrt(eig_val[1])
	semimaj *= scale
	semimin *= scale

	phi = np.arccos(np.dot(eig_vec[0], np.array([1, 0])))
	if eig_vec[0][1] < 0 and phi > 0:
		phi *= -1

	# Generate data for ellipse structure
	theta = np.linspace(0, 2 * np.pi, theta_num)
	r = 1 / np.sqrt((np.cos(theta)) ** 2 + (np.sin(theta)) ** 2)
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	data = np.array([x, y])
	S = np.array([[semimaj, 0], [0, semimin]])
	R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
	T = np.dot(R, S)
	data = np.dot(T, data)
	data[0] += offset[0]
	data[1] += offset[1]

	# Plot!
	return_fig = False
	if axis is None:
		axis = plt.gca()

	if plot_kwargs is None:
		p, = axis.plot(data[0], data[1], color='r', linestyle='-')
	else:
		p, = axis.plot(data[0], data[1], **plot_kwargs)

	if fill == True:
		if fill_kwargs is None:
			fill_kwargs = dict()
		axis.fill(data[0], data[1], alpha=0.2, color='r')

