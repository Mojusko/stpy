import argparse
import sys
import tempfile
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
	m,n = [(ss-1.)/2. for ss in shape]
	y,x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh
	return h

def gaussian_filters(shape, sigmas):
	(height,width,enter,leave) = shape
	G = np.zeros(shape = shape)
	for q in range(enter):
		for index,val in enumerate(sigmas):
			G[:,:,q,index] = matlab_style_gauss2D(shape = (height,width),sigma = val)
	return G

def gaussian_filters_tf(shape, sigmas):
	G = gaussian_filters(shape, sigmas)
	return tf.cast(tf.Variable(G), tf.float32)



def deepnn(x, initialization_params, no_filters_1 = 32, no_filters_2 = 64):

		"""deepnn builds the graph for a deep net for classifying digits.
		Args:
				x: an input tensor with the dimensions (N_examples, 784), where 784 is the
				number of pixels in a standard MNIST image.
		Returns:
				A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
				equal to the logits of classifying the digit into one of 10 classes (the
				digits 0-9). keep_prob is a scalar placeholder for the probability of
				dropout.
		"""


		# Reshape to use within a convolutional neural net.
		# Last dimension is for "features" - there is only one here, since images are
		# grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
		with tf.name_scope('reshape'):
				x_image = tf.reshape(x, [-1, 28, 28, 1])

		# First convolutional layer - maps one grayscale image to 32 feature maps.
		with tf.name_scope('conv1'):
				#W_conv1 = weight_variable([5, 5, 1, no_filters_1])
				W_conv1 = gaussian_filters_tf([5, 5, 1, no_filters_1],initialization_params[0:no_filters_1])
				b_conv1 = bias_variable([no_filters_1])
				h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

		# Pooling layer - downsamples by 2X.
		with tf.name_scope('pool1'):
				h_pool1 = max_pool_2x2(h_conv1)

		# Second convolutional layer -- maps 32 feature maps to 64.
		with tf.name_scope('conv2'):
				#W_conv2 = weight_variable([5, 5, no_filters_1, no_filters_2])
				W_conv2 = gaussian_filters_tf([5, 5, no_filters_1, no_filters_2],initialization_params[no_filters_1:])
				b_conv2 = bias_variable([no_filters_2])
				h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

		# Second pooling layer.
		with tf.name_scope('pool2'):
				h_pool2 = max_pool_2x2(h_conv2)

		# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
		# is down to 7x7x64 feature maps -- maps this to 1024 features.
		with tf.name_scope('fc1'):
				W_fc1 = weight_variable([7 * 7 * no_filters_2, 1024])
				b_fc1 = bias_variable([1024])

				h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*no_filters_2])
				h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# Dropout - controls the complexity of the model, prevents co-adaptation of
		# features.
		with tf.name_scope('dropout'):
				keep_prob = tf.placeholder(tf.float32)
				h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		# Map the 1024 features to 10 classes, one for each digit
		with tf.name_scope('fc2'):
				W_fc2 = weight_variable([1024, 10])
				b_fc2 = bias_variable([10])

				y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		return y_conv, keep_prob


def conv2d(x, W):
		"""conv2d returns a 2d convolution layer with full stride."""
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
		"""max_pool_2x2 downsamples a feature map by 2X."""
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
																								strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
		"""weight_variable generates a weight variable of a given shape."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)


def bias_variable(shape):
		"""bias_variable generates a bias variable of a given shape."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)


def train_network(mnist, verbose = True,  initialization_params = None, min_steps_val = 10,
 val_size = 3000, dropout = 0.5, learning_rate = 10e-4, maxiter = 500, val_count = 1, batch_size = 80, **kwargs):
	# Import data
	# Create the model
	x = tf.placeholder(tf.float32, [None, 784])

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 10])

	# Build the graph for the deep net
	y_conv, keep_prob = deepnn(x, initialization_params)

	with tf.name_scope('loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
	
	cross_entropy = tf.reduce_mean(cross_entropy)

	with tf.name_scope('adam_optimizer'):
			train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
			correct_prediction = tf.cast(correct_prediction, tf.float32)
	accuracy = tf.reduce_mean(correct_prediction)

	#graph_location = tempfile.mkdtemp()
	#print('Saving graph to: %s' % graph_location)
	#train_writer = tf.summary.FileWriter(graph_location)
	#train_writer.add_graph(tf.get_default_graph())

	init = tf.initialize_all_variables()

	with tf.Session() as sess:
		sess.run(init)
		#sess.run(tf.global_variables_initializer())

		oldval_scores = np.zeros((min_steps_val))
		j = 0
		for i in range(maxiter):
			batch = mnist.train.next_batch(batch_size)
			train_accuracy = accuracy.eval(feed_dict={
									x: batch[0], y_: batch[1], keep_prob: 1.0})
			if i % val_count ==0:

				val_accuracy    = accuracy.eval(feed_dict={
										x: mnist.validation.images[0:val_size],
										 y_: mnist.validation.labels[0:val_size], keep_prob: 1.0})
				oldval_scores[j % min_steps_val] = val_accuracy
				j = j+1
				if verbose==True:
					print('step %d, training accuracy: %f, validation accuracy: %f' % (i, train_accuracy,val_accuracy))
				## validation stopping 
				if i>min_steps_val:
					if np.mean(oldval_scores)>val_accuracy:
						if verbose==True:
							print ("Validation stopping")
						break
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout})

		test_accuracy = accuracy.eval(feed_dict={	x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
		if verbose==True:
			print('test accuracy %g' % test_accuracy)

		return (i,test_accuracy)

if __name__ == "__main__":
	N = 16
	N2 = 32
	sigmas = np.random.randn(N+N2)
	sigmas = sigmas**2
	#print (train_network("/tmp/tensorflow", dropout = 0.7, verbose = False, val_size = 1,  initialization_params = sigmas, no_filters_1=N, no_filters_2=N2))


	##############################################
	########  Visualization of Filters ###########
	##############################################
	import matplotlib as mpl
	V = gaussian_filters((5,5,1,N), sigmas[0:N] )
	fig, axes = plt.subplots(nrows=4, ncols=int(N/4))
	for index,ax in enumerate(axes.flat):
			im = ax.imshow(V[:,:,0,index] ,interpolation='nearest',vmin=0, vmax=1)
	cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
	plt.colorbar(im, cax=cax, **kw)		

	V2 = gaussian_filters((5,5,N,N2), sigmas[N:] )
	fig, axes = plt.subplots(nrows=8, ncols=int(N2/8))
	for index,ax in enumerate(axes.flat):
			im = ax.imshow(V2[:,:,0,index] ,interpolation='nearest',vmin=0, vmax=1)
	cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
	plt.colorbar(im, cax=cax, **kw)
	plt.show()

