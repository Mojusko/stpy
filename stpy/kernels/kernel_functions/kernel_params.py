
class KernelParams():

	def __init__(self, param_dict):
		for key in param_dict:
			setattr(self, key, param_dict[key])

	def assert_existence(self, names):
		for name in names:
			if not hasattr(self, name):
				raise AttributeError("Missing attribute of the kernel %s" % str(name))
