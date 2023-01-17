import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler


def parallel_coordinates_bo(X, Y, names=None, scaling=None, fig_size=(20, 10)):
	"""
		Parallel plot graph

		X : 2D numpy array of parameters [points,parameters]
		Y : 1D numpy array of values
		names: list of names size of (parameters)
		scaling:
			"stat": statistical scaling
			None : no scaling
			(low,hig): tuple, scales to [-1,1]
		fig_size: fig size in inches
	"""

	if scaling == "stat":
		scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
		scaler.fit(X)
		Z = scaler.transform(X)
	elif scaling is None:
		Z = X
	else:
		try:
			Z = X
			up, low = scaling
			d = X.shape[1]
			for i in range(d):
				Z[:, i] = (2 * X[:, i]) / (up[i] - low[i]) + (1.0 - 2 * up[i] / (up[i] - low[i]))
		except:
			pass

	D = np.append(Z, Y, axis=1)
	data = pd.DataFrame(D)
	data = data.sort_values(by=Z.shape[1])
	names = copy.copy(names)
	names.append(Z.shape[1])
	if names is not None:
		data.columns = names
	plt.figure(figsize=(fig_size))
	plt.xticks(rotation=45)
	ax = parallel_coordinates(data, Z.shape[1], colormap="summer")
	ax.get_legend().remove()
	plt.show()


if __name__ == "__main__":
	from stpy.test_functions.protein_benchmark import ProteinBenchmark

	Benchmark = ProteinBenchmark("protein_data_gb1.h5", dim=3, ref=['A', 'B', 'C', 'D'])
	names = Benchmark.data['P1'].values
	Benchmark.self_translate()
	vals = Benchmark.data['P1'].values

	print(Benchmark.data)
	X = Benchmark.data.values[0:8000, 0:3]
	Y = Benchmark.data.values[0:8000, 5].reshape(-1, 1)
	print(X.shape, Y.shape)
	names = ["P1", "P2", "P3"]
	# plt.yticks(vals, names)
	parallel_coordinates_bo(X, Y, names=names)

	plt.show()
