import numpy as np
from sample import Sample


def one_point_crossover(p1: Sample, p2: Sample):
	cutting_point = np.random.randint(1, len(p1.x) - 2)

	child1 = Sample(p1.func, p1.x[:cutting_point] + p2.x[cutting_point:])
	child2 = Sample(p2.func, p2.x[:cutting_point] + p1.x[cutting_point:])

	return [child1, child2]


def n_points_crossover():
	pass


def uniform_crossover(p1: Sample, p2: Sample):
	x1 = [p1.x[i] if np.random.rand() > 0.5 else p2.x[i] for i in range(len(p1.x))]
	x2 = [p1.x[i] if np.random.rand() > 0.5 else p2.x[i] for i in range(len(p1.x))]

	child1 = Sample(p1.func, x1)
	child2 = Sample(p2.func, x2)

	return [child1, child2]



def crossover(p1, p2, crossover_type='one_point'):
	type2function = {
		'one_point': one_point_crossover,
		'n_point': n_points_crossover,
		'uniform': uniform_crossover,
	}

	return type2function[crossover_type](p1, p2)
