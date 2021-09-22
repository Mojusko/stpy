import scipy
import numpy as np
import torch
import mosek
from stpy.helpers.ellipsoid_algorithms import maximize_on_elliptical_slice
import matplotlib.pyplot as plt
import cvxpy as cp
from stpy.embeddings.packing_embedding import PackingEmbedding
from stpy.borel_set import BorelSet
from stpy.point_processes.poisson import PoissonPointProcess

class RateEstimator():

	def __init__(self):
		pass


	def get_min_max(self):
		basic_sets = self.hierarchy.get_sets_level(self.hierarchy.levels)
		volumes = []
		for index, elementary in enumerate(basic_sets):
			volumes.append(elementary.volume())

		return (np.min(volumes), np.max(volumes))



	def load_data(self, data, times = True):
		self.approx_fit = False

		if len(data) > 0:
			self.approx_fit = False
			phis = []
			observations = []
			self.data = data.copy()
			counts = []
			#times_arr = []

			for sample in data:
				S, obs, dt = sample
				count = torch.Tensor([0])

				if obs is not None:
					if times == True:
						emb = self.packing.embed(obs) * dt
					else:
						emb = self.packing.embed(obs)

					phi = self.packing.integral(S) * dt
					observations.append(emb)
					count = torch.Tensor([emb.size()[0]])
					phis.append(phi.view(1, -1))


					if self.dual == True:
						self.global_dt = dt
						dist_matrix = torch.cdist(obs, self.anchor_points, p = 2)
						for k in range(obs.size()[0]):
							index = torch.argmin(dist_matrix[k,:])
							self.anchor_weights[index] = self.anchor_weights[index] + 1.
				else:
					phi = self.packing.integral(S) * dt
					phis.append(phi.view(1, -1))
				counts.append(count)

			self.counts = torch.cat(counts, dim=0)  # n(A_i)
			self.phis = torch.cat(phis, dim=0)  # integrals of A_i

			if len(observations) > 0:
				self.observations = torch.cat(observations, dim=0)  # \{x_i\}_{i=1}^{n(A_i)}
			else:
				self.observations = None

			if self.feedback == "count-record":
				self.bucketization()

	def add_data_point(self, new_data, times = True):
		self.approx_fit = False

		if self.data is None:
			self.load_data([new_data])
			return

		self.data.append(new_data)

		# update standard form data
		S, obs, dt = new_data
		if obs is not None:

			if times == True:
				emb = self.packing.embed(obs) * dt
			else:
				emb = self.packing.embed(obs)

			phi = self.packing.integral(S).view(1, -1) * dt

			count = torch.Tensor([emb.size()[0]])

			if self.observations is not None:
				self.observations = torch.cat((self.observations, emb), dim=0)
				#self.times = torch.cat((self.times, dt * torch.ones(size=(emb.size()[0],1)).view(-1).double() ))
			else:
				self.observations = emb
				#self.times =  dt * torch.ones(size=(emb.size()[0],1)).view(-1).double()


			if self.dual == True:

				dist_matrix = torch.cdist(obs, self.anchor_points, p=2)
				for k in range(obs.size()[0]):
					index = torch.argmin(dist_matrix[k, :])
					self.anchor_weights[index] += 1.
		else:
			count = torch.Tensor([0])
			phi = self.packing.integral(S).view(1, -1) * dt


		self.phis = torch.cat((self.phis, phi), dim=0)
		self.counts = torch.cat((self.counts, count))

		if self.feedback == "count-record":

			for index, elementary in enumerate(self.basic_sets):

				if S.inside(elementary) == True:
					if obs is not None:
						mask = elementary.is_inside(obs)
						self.total_bucketized_obs[index] += float(obs[mask].size()[0])
					else:
						self.total_bucketized_obs[index] += 0.0

					self.bucketized_counts[index] += 1
					self.total_bucketized_time[index] += dt
