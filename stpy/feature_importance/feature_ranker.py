import torch
import numpy as np
from typing import Any
from stpy.estimator import Estimator
import copy

class FeatureRanker():

	def __init__(self,
				 model: Estimator,
				 mode: str  = 'explained variance'
				 ):
		self.model = model
		self.mode = mode

		if not hasattr(self.model, "kernel_object"):
			print ("Invalid estimator structure to run feature importance analysis")

	def importance(self):

		if self.mode == 'explained variance':
			return self.one_off_importance()
		elif self.mode == 'cross_validation':
			raise NotImplementedError("This is not implemented.")

	def one_off_importance(self):
		n,d = self.model.x.size()
		x = self.model.x
		y = self.model.y
		# iterate over features and
		importance = torch.zeros(size=(d,1)).double().view(-1)
		res_total = torch.sum(self.model.residuals(x, y) ** 2)

		for i in range(d):
			# define new data
			xnew = x.clone()
			xnew[:,i] = 0.

			# define new model
			GP = copy.deepcopy(self.model)
			GP.fit_gp(xnew,y)

			# evaluate residuals
			res = torch.sum(GP.residuals(xnew,y)**2)

			# store
			importance[i] = res_total/res
			print(i + 1, "/", d,':', res_total/res)
		return importance