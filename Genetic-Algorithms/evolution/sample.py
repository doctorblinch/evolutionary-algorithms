from collections import namedtuple

# Sample = namedtuple('Sample', 'x fitness')
import numpy as np


class Sample:
	def __init__(self, func, x=None):
		self.func = func

		self.x = x or np.random.randint(0, 2, size=func.meta_data.n_variables).tolist()
		self.x = np.array(self.x)
		self.fitness = self.func(self.x)

	def __repr__(self):
		return f'<Sample {self.fitness}>'
