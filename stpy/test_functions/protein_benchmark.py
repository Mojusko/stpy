from stpy.test_functions.benchmarks import BenchmarkFunction
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import stpy.helpers.helper as helper

class ProteinOperator():

	def __init__(self):

		self.real_names = {'A':'Ala', 'R':'Arg', 'N':'Asn', 'D':'Asp', 'C':'Cys','Q':'Gln',  'E':'Glu','G':'Gly',
				'H':'His','I':'Iso','L':'Leu',	'K':'Lys','M':'Met','F':'Phe',
				'P':'Pro','S':'Ser','T':'Thr','W':'Trp','Y':'Tyr','V':'Val','B':'Asx'}

		self.dictionary = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4,'Q':5,'E':6,'G':7,
				'H':8,'I':9,'L':10,	'K':11,'M':12,'F':13,
				'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19,'B':3}

		self.inv_dictionary  = {v: k for k, v in self.dictionary.items()}


		self.Negative = ['D', 'E']
		self.Positive = ['R', 'K', 'H']
		self.Aromatic = ['F', 'W', 'Y','H']
		self.Polar = ['N', 'Q', 'S', 'T','Y']
		self.Aliphatic = ['A','G','I','L','V']
		self.Amide = ['N','Q']
		self.Sulfur = ['C','M']
		self.Hydroxil = ['S','T']
		self.Small = ['A', 'S', 'T', 'P', 'G', 'V']
		self.Medium = ['M', 'L', 'I', 'C', 'N', 'Q', 'K', 'D', 'E']
		self.Large = ['R', 'H', 'W', 'F', 'Y']
		self.Hydro = ['M', 'L', 'I', 'V', 'A']
		self.Cyclic = ['P']
		self.Random = ['F', 'W', 'L', 'S', 'D']


	def translate(self,X):
		f = lambda x: self.dictionary[x]
		Y = X.copy()
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				Y[i,j] = f(X[i,j])
		return Y.astype(int)

	def remove_wild_type_mutations(self,mutation):
		mutation_split = mutation.split("+")
		output = []
		for mut in mutation_split:
			if mut[0] != mut[-1]:
				output.append(mut)
		return "+".join(output)

	def get_substitutes_from_mutation(self, mutation):
		mutation_split = mutation.split("+")
		original = []
		new = []
		positions = []

		for mut in mutation_split:
			original.append(mut[0])
			new.append(mut[-1])
			positions.append(int(mut[1:-1]))

		return (original,new,positions)

	def mutation(self, original_seq, positions, new_seq):
		old_seq = list(original_seq)
		new_seq = list(new_seq)
		identifier = []
		for old, new, position in zip(old_seq, new_seq, positions):
			if old != new:
				identifier.append(old + str(position) + new)
		return '+'.join(identifier)


	def translate_amino_acid(self,letter):
		return self.dictionary[letter]

	def translate_one_hot(self,X):
		try:
			Y = self.translate(X)
		except:
			Y = X
		n,d = list(X.shape)
		Z = np.zeros(shape=(n,d*self.total))
		for i in range(n):
			for j in range(d):
				Z[i,Y[i,j]+j*self.total] = 1.0

		return Z

	def get_real_name(self, name):
		out = []
		for i in name:
			out.append(self.real_names[i])
		return out

class ProteinBenchmark(BenchmarkFunction):

	def __init__(self,fname, dim = 1, ref = ['D','D','D','D'], avg = False, scale = True):
		"""
		initialize the protein benchmark

		 fname : dataset name
		 dim : dimension of the dataset
		 ref : for smaller dimensions what is the reference in the 4 dim space?
		 avg : average the effect over other combinations in lower dimensions
		"""


		"""
		Convention of the following dictionary is to map B->D as B can stand for N and D. 
		"""

		self.dictionary = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4,'Q':5,'E':6,'G':7,
				'H':8,'I':9,'L':10,	'K':11,'M':12,'F':13,
				'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19,'B':3}

		f = lambda x: self.dictionary[x]

		self.total = 20
		self.dim = dim
		self.ref = ref
		self.ref_translated = [f(x) for x in self.ref]

		dset = pd.read_hdf(fname)

		# average the effect over others
		if avg == False:
			mask = np.full(dset.shape[0], True, dtype=bool)
			for j in range(4-dim):
				mask = np.logical_and(mask,dset["P" + str(4-j)] == ref[3-j])
			self.data = dset[mask]
		else:
			# avg. not implemented
			pass

		if scale == True:
			maximum = np.max(self.data[:]['Fitness'])
			self.data[:]['Fitness']=self.data[:]['Fitness']/maximum
		else:
			pass


		self.real_names = {'A':'Ala', 'R':'Arg', 'N':'Asn', 'D':'Asp', 'C':'Cys','Q':'Gln',  'E':'Glu','G':'Gly',
				'H':'His','I':'Iso','L':'Leu',	'K':'Lys','M':'Met','F':'Phe',
				'P':'Pro','S':'Ser','T':'Thr','W':'Trp','Y':'Tyr','V':'Val','B':'Asx'}


		self.Negative = ['D', 'E']
		self.Positive = ['R', 'K', 'H']
		self.Aromatic = ['F', 'W', 'Y','H']
		self.Polar = ['N', 'Q', 'S', 'T','Y']
		self.Aliphatic = ['A','G','I','L','V']
		self.Amide = ['N','Q']
		self.Sulfur = ['C','M']
		self.Hydroxil = ['S','T']
		self.Small = ['A', 'S', 'T', 'P', 'G', 'V']
		self.Medium = ['M', 'L', 'I', 'C', 'N', 'Q', 'K', 'D', 'E']
		self.Large = ['R', 'H', 'W', 'F', 'Y']
		self.Hydro = ['M', 'L', 'I', 'V', 'A']
		self.Cyclic = ['P']
		self.Random = ['F', 'W', 'L', 'S', 'D']

	def get_real_name(self, name):
		out = []
		for i in name:
			out.append(self.real_names[i])
		return out


	def data_summary(self):
		y = self.data['Fitness'].values
		maximum = np.max(y)
		minimum = np.min (y)
		return (maximum,minimum)


	def translate(self,X):
		f = lambda x: self.dictionary[x]
		Y = np.zeros(shape = X.shape).astype(int)
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				Y[i,j] = f(X[i,j])
		return Y

	def translate_one_hot(self,X):
		try:
			Y = self.translate(X)
		except:
			Y = X
		n,d = list(X.shape)
		Z = np.zeros(shape=(n,d*self.total))
		for i in range(n):
			for j in range(d):
				Z[i,Y[i,j]+j*self.total] = 1.0

		return Z

	def self_translate(self):
		"""
		self translate from
		:return:
		"""
		f = lambda x: self.dictionary[x]
		for j in range(4):
			self.data['P'+str(j+1)] = self.data['P'+str(j+1)].apply(f)

	def set_fidelity(self,F):
		self.Fidelity = F

	def scale(self):
		self.scale = 1

	def eval_noiseless(self,X):
		"""
		evaluate depends on the dimension
		"""
		res = []

		# append
		n = X.shape[0]
		C = np.tile(self.ref_translated[self.dim:4],(n,1))
		X_ = np.concatenate((X,C),axis = 1)
		for i in range(n):
			x = X_[i,:]
			mask = np.full(self.data.shape[0], True, dtype=bool)
			for j in range(4):
				#print (x[j],self.data["P" + str(j + 1)])
				mask = np.logical_and(mask, self.data["P" + str(j + 1)] == x[j])
			res.append(self.data[mask]['Fitness'].values)
		return np.array(res).reshape(-1,1)

	def interval_number(self, dim = None):
		if dim is None:
			dim = self.dim
		arr = self.interval_letters(dim = dim)
		out = self.translate(arr)
		return out

	def interval_onehot(self, dim = None):
		if dim is None:
			dim = self.dim
		arr = self.interval_letters(dim = dim)
		out = self.translate_one_hot(arr)
		return out

	def interval_letters(self, dim = None):
		if dim is None:
			dim = self.dim

		names = list(self.dictionary.keys())
		names.remove('B')
		arr = []
		for i in range(dim):
			arr.append(names)
		out = helper.cartesian(arr)
		return out

	# def actions(self):
	# 	number_of_actions = self.dim*(20**(self.dim-1))
	#
	# 	actions = []
	#
	# 	## this includes (20,d) actions
	# 	one_dim = self.interval_onehot(dim = 1)
	# 	#print (one_dim)
	# 	#print ("one dim",one_dim.shape)
	# 	if self.dim - 1>0:
	# 		# this includes (20**(d-1), d) actions
	# 		others = self.interval_onehot(dim = self.dim - 1)
	# 		#print ("others:", others.shape)
	# 		for fix_dim in range(self.dim):
	# 			#print (fix_dim)
	# 			action = np.zeros(shape=(20 ** (self.dim - 1), 20 * self.dim))
	# 			for elem in one_dim:
	# 				#print (fix_dim*20+(fix_dim+1)*20)
	# 				action[:,fix_dim*20:(fix_dim+1)*20]=elem
	# 				action[:,0:fix_dim*20] = others[:,0:fix_dim*20]
	# 				action[:,(fix_dim+1) * 20:] = others[:,fix_dim*20:]
	# 				actions.append(action)
	# 		return actions
	# 	else:
	# 		return one_dim


	def actions(self):
		number_of_actions = self.dim*(20**(self.dim-1))

		actions = []

		## this includes (20,d) actions
		one_dim = self.interval_onehot(dim = 1)
		#print (one_dim)
		#print ("one dim",one_dim.shape)
		if self.dim - 1>0:
			# this includes (20**(d-1), d) actions
			others = self.interval_onehot(dim = self.dim - 1)
			#print ("others:", others.shape)
			for elem in others:
				for fix_dim in range(self.dim):
					action = np.zeros(shape=(20, 20 * self.dim))
					action[:,fix_dim*20:(fix_dim+1)*20]=one_dim
					j = 0
					for i in range(self.dim):
						if i != fix_dim:
							action[:,i*20:(i+1)*20] = elem[j*20:(j+1)*20]
							j = j + 1

					actions.append(action)
			return actions
		else:
			return one_dim

	def subsample_dts_indice_only(self, N, split = 0.9):
		self.self_translate()
		xtest = self.interval_onehot()

		indices = np.arange(0,N,1)
		sample = indices
		np.random.shuffle(indices)

		train = sample[0:int(np.round(split*N))]
		test = sample[int(np.round(split*N)):N]

		return (train, test)
	def subsample_dts(self, N, split = 0.90):
		self.self_translate()
		xtest = self.interval_onehot()
		indices = np.arange(0,N,1)

		indices = np.random.shuffle(indices)
		sample = xtest[indices,:]

		y_sample = self.eval_one_hot(sample)

		x_train = sample[0:int(np.round(split*N)),:]
		y_train = y_sample[0:int(np.round(split*N)),:]
		x_test = sample[int(np.round(split*N)):N,:]
		y_test = y_sample[int(np.round(split*N)):N,:]

		return (x_train,y_train,x_test,y_test)


	def eval_fidelity(self,X):
		return self.Fidelity(X)

	def eval(self,X):
		z = self.eval_noiseless(X)
		return z

	def eval_one_hot(self,X):
		n, d = list(X.shape)
		Z = np.zeros(shape=(n, self.dim ))
		for i in range(n):
			for j in range(d):
				if 	X[i,  j] > 0:
					Z[i,j // self.total] = j % self.total
		Z = Z.astype(int)
		Y = self.eval(Z)
		return Y

	def plot_one_site_map(self,kernel, save = None, dim = 1):
		plt.figure()
		names = list(self.dictionary.keys())
		names.remove('B')
		real_names = self.get_real_name(names)
		real_names = helper.cartesian([real_names for i in range(dim)])


		xtest = torch.from_numpy(self.interval_onehot(dim = dim))
		real_names = [ ','.join(list(i)) for i in real_names]
		ax = plt.imshow(kernel(xtest, xtest).detach().numpy())
		plt.colorbar()
		plt.xticks(range(xtest.shape[0]),real_names,fontsize=10, rotation= 60)
		plt.yticks(range(xtest.shape[0]),real_names,fontsize=10)
		plt.margins(0.2)
		if save is not None:
			plt.savefig(save)
		else:
			plt.show()

if __name__ == "__main__":
	Benchmark = ProteinBenchmark("protein_data_gb1.h5", dim = 2, ref = ['A','B','C','D'])
	#print (Benchmark.data)
	Benchmark.self_translate()
	Benchmark.data.plot.scatter(x='P1', y='P2', c=Benchmark.data['Fitness'], s = 200)
	#print (Benchmark.data)
	X = np.array([['F','C'],['D','C']])
	X_ = Benchmark.translate(X)
	print (X,X_)
	X__ = Benchmark.translate_one_hot(X)

	print (Benchmark.translate_one_hot(X))

	print (Benchmark.eval(X_))

	print (Benchmark.eval_one_hot(X__))