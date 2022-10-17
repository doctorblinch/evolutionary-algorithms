import numpy as np


def matting_selection(population, fitnesses):
	min_fit = np.min(fitnesses)
	sum_fit = np.sum(fitnesses)

	ps = (fitnesses - min_fit) / (sum_fit - fitnesses.shape[0] * min_fit)
	ps = np.nan_to_num(ps, copy=False)
	if np.sum(ps) == 0:
		ps += (1-np.sum(ps))/(ps.shape[0])
		# print(sum(ps))
	return population[np.random.choice(population.shape[0], size=population.shape[0], p=ps)]


def one_point_crossover(p1, p2, crossover_prob):
	if np.random.rand() > crossover_prob:
		return [p1, p2]

	cutting_point = np.random.randint(1, p1.shape[0] - 1)
	child1 = np.concatenate((p1[:cutting_point], p2[cutting_point:]))
	child2 = np.concatenate((p2[:cutting_point], p1[cutting_point:]))

	return [child1, child2]


def n_point_crossover(p1, p2, crossover_prob, n):
	for i in range(n):
		p1, p2 = one_point_crossover(p1, p2, crossover_prob)

	return [p1, p2]


def crossover(population, crossover_prob=0.3, crossover_type=1):
	res = []

	if crossover_type == 1:
		for i in range(0, population.shape[0], 2):
			res += one_point_crossover(population[i], population[i + 1], crossover_prob)

		return np.array(res)

	if crossover_type == -1:
		for i in range(0, population.shape[0], 2):
			for j in range(population[0].shape[0]):
				if np.random.rand() < crossover_prob:
					population[i], population[i+1] = population[i+1], population[i]

		return population

	for i in range(0, population.shape[0], 2):
		res += n_point_crossover(population[i], population[i + 1], crossover_prob, crossover_type)

	return np.array(res)


def mutate(population, mutation_prob=0.1, mutation_type='flip', iteration2budget=None):
	if mutation_type == 'flip':
		return np.logical_xor(population, (np.random.uniform(size=population.shape) < mutation_prob)).astype(int)

	if mutation_type == 'swap':
		for sample in population:
			for i in range(sample.shape[0]):
				if np.random.rand() < mutation_prob:
					pos2 = np.random.randint(0, sample.shape[0])
					sample[i], sample[pos2] = sample[pos2], sample[i]

		return population

	if mutation_type == 'smart':
		prob = mutation_prob * (0.1 - 0.09) / iteration2budget
		return np.logical_xor(population, (np.random.uniform(size=population.shape) < prob)).astype(int)
