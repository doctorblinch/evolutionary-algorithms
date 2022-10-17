import os
from multiprocessing import Pool, cpu_count, Process

from ioh import get_problem
from ioh import logger

from mating_selection import *
from crossover import *
from mutation import *
from sample import Sample

POPULATION_SIZE = 100
SELECTION_SIZE = 100
MUTATION_PROB = 0.05


def evaluation(population: List[Sample]) -> Tuple[List, float]:
	f_opt = max(population, key=lambda i: i.fitness).fitness
	x_opt = [population[i].x for i in range(len(population)) if population[i].fitness == f_opt][0]

	return x_opt, f_opt


def inpopulation_difference(population: List[Sample]) -> float:
	res = []

	for i in population:
		for j in population:
			if i is j:
				continue

			res.append(sum(np.abs([i.x[k]-j.x[k] for k in range(len(i.x))])))

	return float(np.mean(res))


def eva(func, budget=None, mutation_prob=MUTATION_PROB, random_state=None):
	if random_state is not None:
		np.random.seed(random_state)

	if budget is None:
		budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

	if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
		optimum = 8
	else:
		optimum = func.objective.y

	population = [Sample(func) for i in range(POPULATION_SIZE)]

	x_opt, f_opt = evaluation(population)

	for i in range(budget):
		if i % 500 == 0:
			print(f'Iteration {i} f_opt={f_opt}, inpopulation distance={inpopulation_difference(population)}')

		# Selection
		selection_pair = [proportional_selection(population, size=2) for _ in range(SELECTION_SIZE)]
		# Crossover
		children = []

		for pair in selection_pair:
			children += crossover(*pair, crossover_type='uniform')

		# Mutation
		for child in children:
			mutate(child, mutation_prob)

		# Reduce population
		population += children
		population = proportional_selection(population, size=POPULATION_SIZE)
		x_opt, f_opt = evaluation(population)

	return [f_opt, x_opt]


if __name__ == '__main__':
	# Declaration of problems to be tested.
	om = get_problem(1, dim=50, iid=1, problem_type='PBO')
	lo = get_problem(2, dim=50, iid=1, problem_type='PBO')
	labs = get_problem(18, dim=32, iid=1, problem_type='PBO')

	l = logger.Analyzer(root="data", folder_name="run", algorithm_name="random_search",
	                    algorithm_info="test of IOHexperimenter in python")

	om.attach_logger(l)
	res = None
	parallel = 4
	mutation_prob = 0.3

	if parallel:
		for i in range(parallel):
			p = Process(target=eva, args=(om, None, mutation_prob, i)).start()
		p.join()
	else:
		print(eva(om, mutation_prob=mutation_prob))

	print(res)

	lo.attach_logger(l)
	labs.attach_logger(l)

	del l
