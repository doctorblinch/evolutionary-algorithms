import json
import os.path
import random
import sys
from time import time
from typing import List
import multiprocessing as mp

import pandas as pd
from ioh import get_problem
from ioh import logger
import numpy as np

from recombination import recombine
from mutation import mutate
from selection import select_population

# Parameters
POPULATION_SIZE = 150
OFFSPRING_SIZE = 150
RECOMBINATION_TYPE = 'intermediate'  # intermediate | discrete
SELECTION_TYPE = 'plus'  # coma | plus
BUDGET = 50_000
DIM = 5
INIT_SCALER = 2.0
l = None

IOH_FOLDER = 'experiment'
CSV_FOLDER = 'experiment_csv'

T = 1 / np.sqrt(2 * np.sqrt(DIM))
T0 = 1 / np.sqrt(2 / DIM)


def mse(true_res, res):
	return (true_res - res)**2


def mae(true_res, res):
	return abs(true_res - res) / abs(true_res)


def ES(func, budget=BUDGET, population_size=POPULATION_SIZE, offspring_size=OFFSPRING_SIZE,
       recombination_type=RECOMBINATION_TYPE, selection_type=SELECTION_TYPE
       ):
	start_time = time()
	parent = ((np.random.rand(POPULATION_SIZE * func.meta_data.n_variables) - 0.5) * INIT_SCALER) \
		.astype(np.float32) \
		.reshape(POPULATION_SIZE, func.meta_data.n_variables)

	parent_fitness = np.array([func(par) for par in parent])  # Evaluate parent
	budget -= POPULATION_SIZE
	mutation_params = np.abs(np.random.normal(0, T0, (population_size, func.meta_data.n_variables)))

	while budget > 0:
		# Generate offspring and evaluate
		# Note that this is just an example, and there may be difference among different evolution strategies.
		# You shall implement your own evolution strategy.
		offspring = recombine(parent, offspring_size, rec_type=recombination_type)
		offspring, mutation_params = mutate(offspring, mutation_params)
		offpsring_fitness = np.array([func(par) for par in offspring])  # Evaluate offspring
		budget = budget - offspring_size
		parent, parent_fitness = select_population(parent, offspring, parent_fitness, offpsring_fitness, select_type='plus')

		# if budget % 5_000 == 0:

	processing_time = time() - start_time
	metrics = mae(parent_fitness.min(), func.objective.y)

	print(parent_fitness.round(2).min(), round(func.objective.y, 2),
	      np.isclose(parent_fitness.round(2).min(), round(func.objective.y, 2), atol=1e-3),
	      metrics)

	return {'mae': metrics, 'time': processing_time}


def task(pid, **kwargs):
	results = []
	# Testing 10 instances for each problem
	for ins in range(1, 11):
		# Getting the problem with corresponding problem_id, instance_id, and dimension n = 5.
		problem = get_problem(pid, dim=DIM, iid=ins, problem_type='BBOB')

		# Attach the problem with the logger
		problem.attach_logger(l)

		# The assignment 2 requires only one run for each instance of the problem.
		es_res = ES(problem, **kwargs)
		es_res.update({
			'pid': pid,
			'iid': ins,
			'dim': DIM
		})

		results.append(es_res)

		# To reset evaluation information as default before the next independent run.
		# DO NOT forget this function before starting a new run. Otherwise, you may meet issues with uploading data to IOHanalyzer.
		problem.reset()

	return results


def main(file_name, single_thread=False):
	global l
	l = logger.Analyzer(root=IOH_FOLDER,
	                    folder_name="run",
	                    algorithm_name="s3069176",
	                    algorithm_info="An Evolution Strategy on the 24 BBOB problems in python")

	if not os.path.exists(CSV_FOLDER):
		os.mkdir(CSV_FOLDER)

	# Testing on 24 problems
	res = []

	# try:
	if single_thread:
		for i in range(1, 25):
			res.append(task(i))
	else:
		with mp.Pool(mp.cpu_count()) as p:
			res = p.map(task, list(range(1, 25)))

	df = pd.DataFrame([j for i in res for j in i])
	with open(os.path.join(CSV_FOLDER, file_name[:-3] + 'json'), 'w') as f:
		json.dump({
			'mae_mean': df['mae'].mean(),
			'POPULATION_SIZE': POPULATION_SIZE,
			'OFFSPRING_SIZE': OFFSPRING_SIZE,
			'RECOMBINATION_TYPE': RECOMBINATION_TYPE,
			'SELECTION_TYPE': SELECTION_TYPE
		}, f)


	df.to_csv(os.path.join(CSV_FOLDER, file_name), index=False)
	# This statemenet is necessary in case data is not flushed yet.

	del l
	# except:
	# 	pass


def random_search(parameters: dict, n_samples: int):
	for _ in range(n_samples):
		yield {k: random.sample(v, 1)[0] for k, v in parameters.items()}


def read_from_config(json_path):
	conf = None

	with open(json_path, 'r') as f:
		conf = json.load(f)
	global POPULATION_SIZE, OFFSPRING_SIZE, RECOMBINATION_TYPE, SELECTION_TYPE
	POPULATION_SIZE = conf['POPULATION_SIZE']
	OFFSPRING_SIZE = conf['OFFSPRING_SIZE']
	RECOMBINATION_TYPE = conf['RECOMBINATION_TYPE']
	SELECTION_TYPE = conf['SELECTION_TYPE']


if __name__ == '__main__':
	# Create default logger compatible with IOHanalyzer
	# `root` indicates where the output files are stored.
	# `folder_name` is the name of the folder containing all output. You should compress this folder and upload it to IOHanalyzer
	params_space = {
		'POPULATION_SIZE': np.round(np.logspace(1, 3, 15)).astype(int).tolist(),
		'OFFSPRING_SIZE': np.round(np.logspace(1, 3, 15)).astype(int).tolist(),
		'RECOMBINATION_TYPE': ['intermediate', 'discrete'],
		'SELECTION_TYPE': ['plus', 'coma']
	}

	# read_from_config('experiment_csv/result-37.json')
	read_from_config('optimal_config.json')
	main('result_ind.csv', True)

	# for index, params in enumerate(random_search(params_space, 50)):
	# 	POPULATION_SIZE = params['POPULATION_SIZE']
	# 	OFFSPRING_SIZE = params['OFFSPRING_SIZE']
	# 	RECOMBINATION_TYPE = params['RECOMBINATION_TYPE']
	# 	SELECTION_TYPE = params['SELECTION_TYPE']
	# 	main(f'result-{index}.csv', single_thread=True)

