from stpy.continuous_processes.fourier_fea import GaussianProcessFF
from stpy.test_functions.benchmarks import *
from doexpy.bandits.OPPR_TS_GP import OPPR_TS_GP


def get_angle(R):
	v = torch.Tensor([1.0,1.0]).double()
	a1 =  np.arccos((torch.dot(v,R@v)/torch.dot(v,v)).numpy())
	a2 =  np.arccos(-(torch.dot(v,R@v)/torch.dot(v,v)).numpy())
	return np.min([a1,a2])


if __name__ == "__main__":
	from stpy.embeddings.embedding import HermiteEmbedding
	N = 1
	s = 0.0001
	n = 20
	L_infinity_ball = 0.5

	d = 2

	thetae = np.radians(35.)
	ce, se = np.cos(thetae), np.sin(thetae)
	R = torch.from_numpy(np.array(((ce, -se), (se, ce))))

	BenchmarkFunc = MichalBenchmark(d = d, R = R)

	x = BenchmarkFunc.initial_guess(N)
	xtest = BenchmarkFunc.interval(n)
	gamma = BenchmarkFunc.bandwidth()
	bounds = BenchmarkFunc.bounds()
	BenchmarkFunc.scale_max(xtest=xtest)

	print ("Gamma:",gamma)

	F = lambda x: BenchmarkFunc.eval(x, sigma=s)
	F0 = lambda x: BenchmarkFunc.eval(x, sigma=0)


	rot_out = open("rotOut.txt",'w')


	m = 64
	GP = GaussianProcessFF(d=d, s=s, m = torch.ones(d)*m, gamma=gamma*torch.ones(d), bounds=bounds, groups = stpy.helpers.helper.full_group(d))
	#GP = GaussianProcess(d =d ,s = s, gamma = gamma*torch.ones(d) ,groups = stpy.helper.full_group(d))
	#GP = GaussianProcess(d=d, s=s, gamma=gamma, groups=None)

	m = 512
	embedding = HermiteEmbedding(gamma=gamma, m=m, d=d, diameter=1, approx = "hermite")
	Map = lambda x: embedding.embed(x)



	x0 = torch.Tensor([0., 0.]).double().view(-1, d)
#	Bandit = OPPR_TS_GP(x0, F, GP, Map, finite_dim=False, s = 10e-8)
	Bandit = OPPR_TS_GP(x0, F, GP, Map, finite_dim=True, s = s, GPMap = True)

	Rep = 2
	Bandit.decolerate(x0,10e-9,Rep)

	print ("True:",thetae)
	print (R)
	print("Angle:",get_angle(R))

	rot_out.write(str(get_angle(R))+"\n")

	print ("E design:\n",Bandit.Q)
	print("Angle:",get_angle(Bandit.Q.detach()))
	rot_out.write(str(get_angle(Bandit.Q.detach()))+"\n")

	# Gaussian Design
	#Design = torch.randn(size = (Nd,d),dtype = torch.float64)*0.1

	Design = Bandit.design
	y = Bandit.value_design
	for repeats in range(5):
		B = Bandit.inverse_sliced_regression(Design,y,slices = Rep)
		print ("Recovered from SRI:\n",B)
		print (get_angle(B))
		rot_out.write(str(get_angle(B)) + " ")

	rot_out.write("\n")
	BB = Bandit.bootstrap_inverse_sliced_regression(Design,y,slices = Rep,repeats = 20)
	print ("Bootstrap",BB)
	rot_out.write(str(get_angle(torch.from_numpy(BB)))+"\n")

	for _ in range(5):
		Bandit.GP2.optimize_params(type="rots", restarts=1)
		print (Bandit.GP2.Rot)
		rot_out.write(str(get_angle(Bandit.GP2.Rot))+" ")
	rot_out.write("\n")
	rot_out.close()
