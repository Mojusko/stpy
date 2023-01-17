import matplotlib.pyplot as plt
import numpy as np
import torch


def get_increment(eta, steps, f, w0, path=False):
	"""

	:param eta: terminal time
	:param steps: number of steps
	:param f: the operator
	:param w0: initial point
	:return:
	"""

	tau = eta / steps
	w = w0
	sequence = []

	for i in range(steps):

		n = torch.randn(size=w0.size()).double()
		w = w + np.sqrt(2 * tau) * f(w, n)
		if path:
			sequence.append(w)

	if path:
		return sequence
	else:
		return w


if __name__ == "__main__":

	f = lambda w: torch.diag(1. / torch.abs(w.view(-1)))
	d = 1
	w0 = torch.zeros(size=(d, 1)).double() + 2
	step = 100
	path = get_increment(2, step, f, w0, path=True)
	# plt.plot(path)

	i = 0
	colors = ['k', 'r', 'b', 'orange', 'brown', 'purple']
	for steps in [5, 10, 20, 100, 200, 500]:

		repeats = 100
		ws = []
		for _ in range(repeats):
			path = get_increment(2, steps, f, w0, path=True)
			xtest = torch.linspace(0, 2, steps)
			plt.plot(xtest, path, color=colors[i])
		i = i + 1
	#	plt.hist(np.array(ws), label = str(step))

	plt.legend()
	plt.show()
