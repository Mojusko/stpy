
"""
Adapation of haarfisz: Software to perform Haar Fisz transforms


"""
import numpy as np


def haar_fisz_transform(data):
	a = 2.
	n = data.shape[0]
	nhalf = n // 2

	J = np.log2(n)
	res = data.copy()
	sm = np.zeros(shape = nhalf,dtype = float)
	det = sm.copy()

	for i in np.arange(0,J,1):
		indices = np.arange(0,nhalf,1)

		sm[0:nhalf] = (res[2 * indices] + res[2 * indices + 1 ])/a
		det[0:nhalf] = (res[2 * indices] - res[2 * indices + 1])/a

		det[sm > 0] = det[sm > 0]/np.sqrt(sm[sm > 0])

		res[0:nhalf] = sm[0:nhalf]
		res[nhalf:n] = det[0:nhalf]

		n = n // 2
		nhalf = nhalf // 2
		sm = np.zeros(shape=nhalf)
		det = sm.copy()

	nhalf = 1
	n = 2
	sm = np.zeros(shape=nhalf)
	det = sm.copy()
	for i in np.arange(0,J,1):

		indices = np.arange(0,nhalf,1)
		sm[indices] = res[indices]
		det[indices] = res[nhalf:n]
		res[2 * indices] = a/2. * (sm[indices] + det[indices])
		res[2 * indices+1] = a/2. * (sm[indices] - det[indices])

		n = 2 * n
		nhalf = 2 * nhalf

		sm = np.zeros(shape=nhalf)
		det = sm.copy()
	return res

def inverse_haar_fisz_transform(data):
	a = 2.
	n = data.shape[0]
	nhalf = n//2
	J = np.log2(n)
	res = data.copy()
	sm = np.zeros(shape = nhalf)
	det = sm.copy()

	for i in np.arange(0,J,1):
		indices = np.arange(0,nhalf,1)

		sm[0:nhalf] = (res[2 * indices] + res[2 * indices+1])/a
		det[0:nhalf] = (res[2 * indices] - res[2 * indices+1])/a
		res[0:nhalf] = sm[0:nhalf]
		res[(nhalf ):n] = det[0:nhalf]
		n = n//2
		nhalf = nhalf//2

	nhalf = 1
	n = 2

	for i in np.arange(0,J,1):
		sm[0:nhalf] = res[0:nhalf]
		det[0:nhalf] = res[nhalf:n]
		indices = np.arange(0,nhalf,1)

		res[2 * indices] = (a/2.) * (sm[0:nhalf] + det[0:nhalf] * np.sqrt(sm[0:nhalf]))
		res[2 * indices+1] = (a/2.) * (sm[0:nhalf] - det[0:nhalf] * np.sqrt(sm[0:nhalf]))
		res[res < 0.] = 0.
		n = 2 * n
		nhalf = 2 * nhalf
	return res


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	s = np.random.poisson(5, 4)*0+1
	s2 = np.random.poisson(20, 4)*0+3
	s = np.concatenate((s,s2)).astype(float)
	plt.plot(s)
	v = haar_fisz_transform(s)
	s_inv = inverse_haar_fisz_transform(v)
	plt.plot(v)
	plt.plot(s_inv,'--')
	plt.show()