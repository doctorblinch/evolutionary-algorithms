from typing import Optional

import numpy as np


def select_population(parent: np.ndarray, offspring: Optional[np.ndarray] = None,
                      parent_fitnes: np.ndarray = None, offpsring_fitness: np.ndarray = None,
                      select_type='coma', mu: int = None):
	if mu is None:
		mu = parent.shape[0]

	if select_type == 'coma':
		return coma_selection(offspring, offpsring_fitness, mu)

	if select_type == 'plus':
		return coma_selection(np.concatenate([parent, offspring]), np.concatenate([parent_fitnes, offpsring_fitness]), mu)


def coma_selection(offspring: Optional[np.ndarray] = None,
                   offpsring_fitness: np.ndarray = None,
                   mu: int = None):
	if mu == offspring.shape[0]: return offspring
	indexes = np.argsort(offpsring_fitness)
	return offspring[indexes][:mu], offpsring_fitness[indexes][:mu]
