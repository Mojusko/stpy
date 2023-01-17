import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import webcolors


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


def closest_colour(requested_colour):
	min_colours = {}
	for name, key in webcolors.css3_hex_to_names.items():
		r_c, g_c, b_c = webcolors.hex_to_rgb(key)
		rd = (r_c - requested_colour[0]) ** 2
		gd = (g_c - requested_colour[1]) ** 2
		bd = (b_c - requested_colour[2]) ** 2
		min_colours[(rd + gd + bd)] = name
	return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
	try:
		closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
	except ValueError:
		closest_name = closest_colour(requested_colour)
		actual_name = None
	return actual_name, closest_name


def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
			  linewidth=3, alpha=1.0):
	"""
	http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
	http://matplotlib.org/examples/pylab_examples/multicolored_line.html
	Plot a colored line with coordinates x and y
	Optionally specify colors in the array z
	Optionally specify a colormap, a norm function and a line width
	"""

	# Default colors equally spaced on [0,1]:
	if z is None:
		z = np.linspace(0.0, 1.0, len(x))

	# Special case if a single number:
	if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
		z = np.array([z])

	z = np.asarray(z)

	segments = make_segments(x, y)
	lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
							  linewidth=linewidth, alpha=alpha)

	ax = plt.gca()
	ax.add_collection(lc)

	return lc


def make_segments(x, y):
	"""
	Create list of line segments from x and y coordinates, in the correct format
	for LineCollection: an array of the form numlines x (points per line) x 2 (x
	and y) array
	"""

	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)
	return segments
