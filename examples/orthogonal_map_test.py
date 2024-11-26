import torch
from stpy.embeddings.random_nn import RandomMap
from stpy.test_functions.protein_benchmark import ProteinBenchmark
from sklearn.model_selection import train_test_split

if __name__ == "__main__":



	dim = 4

	Benchmark = ProteinBenchmark("/home/mojko/Documents/PhD/stpy/stpy/test_functions/protein_data_gb1.h5", dim=dim, ref=['A', 'B', 'C', 'D'])
	Benchmark.self_translate()

	X = Benchmark.data.values[:,0:dim].astype(int)
	Y = Benchmark.data.values[:,5].astype(float).reshape(-1,1)


	X_one_hot = Benchmark.translate_one_hot(X)

	X_train, X_test, y_train, y_test = train_test_split(X_one_hot, Y, test_size = 0.20, random_state = 42)

	X_train = torch.from_numpy(X_train)
	X_test = torch.from_numpy(X_test)
	y_train = torch.from_numpy(y_train)
	y_test = torch.from_numpy(y_test)

	print(X_train.size())
	print(y_train.size())


	print(X_test.size())
	print(y_test.size())

	d = dim*26
	m = dim*26

	ridge = lambda x: torch.relu(x)
	Net = RandomMap(d,m,ridge, output = 1)

	print ("Loss before training: ",Net.loss(X_test,y_test))

	Net.fit_map(X_train,y_train, verbose=1, lr = 10e-1, epochs = 100)

	print ("Net:",Net.forward(X_test[1,:].view(1,-1)))

	print ("Truth:",y_test[1,:])

	print (Net.loss(X_test,y_test))