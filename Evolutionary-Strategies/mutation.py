import numpy as np


def mutate(offspring, mutation_params, T0=(1 / np.sqrt(5)), T=1 / np.sqrt(2 * np.sqrt(5)), mutation_type='individual'):
	if mutation_type == 'individual':
		return mutate_individual(offspring, mutation_params, T0, T)

	if mutation_type == 'correlated':
		return mutate_correlated(offspring, mutation_params, T0, T)


def mutate_correlated(offspring, mutation_params, angels, T0=(1 / np.sqrt(5)), T=1 / np.sqrt(2 * np.sqrt(5)), B=(np.pi / 36)):
	angels += np.random.normal(0, B)

	# Out of boundary correction
	mask = np.abs(angels) > np.pi
	angels[mask] -= 2 * np.pi * np.sign(angels[mask])
	size = offspring.shape[0] * (offspring.shape[0] - 1) / 2

	C = np.array([])
	for i in range(offspring.shape[1] - 1):
		for j in range(i+1, offspring.shape[1]):
			np.append(C, get_rotation_matrix(angels, size, i, j))

	C = np.dot(C, C.T)
	offspring += C
	return offspring, mutation_params, angels


def mutate_individual(offspring, mutation_params, T0=(1 / np.sqrt(5)), T=1 / np.sqrt(2 * np.sqrt(5))):
	mutation_params *= np.exp(
		np.random.normal(0, T0, mutation_params.shape) +
		np.random.normal(0, T,  mutation_params.shape)
	)
	offspring += np.random.normal(0, mutation_params, offspring.shape)
	offspring[offspring > 5] = 5
	offspring[offspring < -5] = -5
	return offspring, mutation_params


def get_rotation_matrix(alpha, size, i, j):
	diag = np.identity(size)
	diag[i, i] = np.cos(alpha)
	diag[i, j] = -np.sin(alpha)
	diag[j, i] = np.sin(alpha)
	diag[j, j] = np.cos(alpha)
	return diag

