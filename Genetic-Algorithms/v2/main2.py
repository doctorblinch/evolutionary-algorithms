import os
import random
from multiprocessing import Pool, cpu_count, Process

from ioh import get_problem
from ioh import logger
import numpy as np
import pandas as pd
from tqdm import tqdm

from functions import *

POPULATION_SIZE = 100
MUTATION_PROB = 0.01
CROSSOVER_PROB = 0.4


def inpopulation_difference(population):
	res = []

	for i in population:
		for j in population:
			if i is j:
				continue

			res.append(sum(np.abs([i[k] - j[k] for k in range(len(i))])))

	return float(np.mean(res))


def eva(func, budget=None, population_size=POPULATION_SIZE,
        crossover_probability=CROSSOVER_PROB, crossover_n=1,
        mutation_type='flip', mutation_probability=MUTATION_PROB,
        random_state=None, parents2new_gen=False, return_params=True):
	if random_state is not None:
		np.random.seed(random_state)

	if budget is None:
		budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50 / population_size)

	if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
		optimum = 8
	else:
		optimum = func.objective.y

	population = np.random.randint(0, 2, size=(POPULATION_SIZE, func.meta_data.n_variables), dtype=np.int8)
	func_vals = np.array([func(i) for i in population])

	index = func_vals.argmax()
	x_opt = population[index]
	f_opt = func_vals[index]

	for index in range(budget):
		if f_opt == func.objective.y:
			break

		new_pop = matting_selection(population, func_vals)
		crossed = crossover(new_pop, crossover_prob=crossover_probability, crossover_type=crossover_n)
		population = mutate(crossed, mutation_probability, mutation_type, iteration2budget=(index + 1) / budget)
		func_vals = np.array([func(i) for i in population])

		if np.max(func_vals) > f_opt:
			f_opt = np.max(func_vals)
			x_opt = population[np.argmax(func_vals)]

	# print(f'Iteration {i} f_opt={f_opt}, inpopulation distance={inpopulation_difference(population)}')
	# if index % 100 == 0:

	if return_params:
		return {'population_size': population_size,
		        'crossover_probability': crossover_probability, 'crossover_n': crossover_n,
		        'mutation_type': mutation_type, 'mutation_probability': mutation_probability,
		        'index': index, 'f_opt': f_opt}

	# print(index, f_opt)
	return [f_opt, x_opt, index]


def random_search(parameters: dict, n_samples: int):
	for _ in range(n_samples):
		yield {k: random.sample(v, 1)[0] for k, v in parameters.items()}


def experiment(problem, params_grid=None, iterations=500, file_name=None):
	if params_grid is None:
		params_grid = {
			'population_size': list(np.logspace(1, 3, 50)),
			'mutation_probability': list(np.logspace(-4, 0, 30)),
			'crossover_probability': list(np.logspace(-4, 0, 30)),
			'crossover_n': [-1] + [1] * 20 + list(range(2, 21)),
			'mutation_type': ['flip', 'swap'] * 3 + ['smart'],
		}

	gen = random_search(params_grid, iterations)

	res = []
	for params in tqdm(gen):
		eva(problem, params)

	if file_name is not None:
		df = pd.DataFrame(res)
		df.sort_values(by=['f_opt', 'index'], ascending=[False, True]).to_csv(file_name, index=False)

	return res


def inspect_config(problem, config, iterations=15):
	return [eva(problem, **config, return_params=False)[::2] for _ in range(iterations)]


if __name__ == '__main__':
	# Declaration of problems to be tested.
	om = get_problem(1, dim=50, iid=1, problem_type='PBO')
	lo = get_problem(2, dim=50, iid=1, problem_type='PBO')
	labs = get_problem(18, dim=32, iid=1, problem_type='PBO')

	l = logger.Analyzer(root="data", folder_name="arun_labs_bs", algorithm_name="LABS_BEST",
	                    algorithm_info="test of IOHexperimenter in python")

	labs.attach_logger(l)
	print(eva(labs, **
	{'population_size': 10, 'mutation_probability': .0303920,
	 'crossover_probability': .2043361, 'crossover_n': 1,'mutation_type': 'flip'}))
	del l
	input('Thats all')

	df = pd.read_csv('results_labs_500.csv')
	# df['evaluations_cnt'] = df['index'] * df['population_size']
	df = df.sort_values(by=['f_opt'], ascending=[False])
	iterations = 15

	res = []
	for i, data in df.drop(['f_opt', 'index'], axis=1).head(20).iterrows():
		d = data.to_dict()
		inspection = inspect_config(labs, d, iterations=iterations)
		res.append([i[0] for i in inspection])
		print(f'Result-{i} got mean score {np.round(np.mean([i[0] for i in inspection]), 2)} in mean iterations = {np.round(np.mean([i[1] for i in inspection]), 2)}.')

	df_res = pd.DataFrame(res)
	df_res.to_csv('rerun_labs_20_15.csv', index=None)


