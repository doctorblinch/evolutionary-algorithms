import numpy as np
from typing import List, Tuple
from sample import Sample


def roulette_wheel_selection(population: List[Sample], size=2) -> List[Sample]:
	population_fitness = sum([i.fitness for i in population])
	chromosome_probabilities = [chorm.fitness / population_fitness for chorm in population]
	return np.random.choice(population, size=size, p=chromosome_probabilities).tolist()


def proportional_selection(population: List[Sample], size=2) -> List[Sample]:
	sf = sum([i.fitness for i in population])
	min_fitness = min(population, key=lambda x: x.fitness).fitness

	ps = [(sample.fitness - min_fitness) / (sf - min_fitness * len(population))
	      if (sf - min_fitness * len(population)) != 0 else 0
	      for sample in population]

	return np.random.choice(population, size=size, p=ps).tolist()



def keep_best(population: List[Sample], size):
	return sorted(population, key=lambda x: x.fitness, reverse=True)[:size]


def selection(pop, scores, k=3):
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k-1):
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]


def mating(fits, pop):
	f_min = min(fits)
	probs = (fits - f_min) / (sum(fits) - fits.size * f_min)
	np.nan_to_num(probs, copy=False)
	sumprobs = sum(probs)
	if sumprobs < 1:
		probs += (1-sumprobs)/len(probs)

	return np.random.choice(a=pop, size=pop.shape[0], replace=True, p = probs)
