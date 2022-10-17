import numpy as np


def recombine(parent: np.ndarray, offspring_cnt=None, rec_type='intermediate') -> np.ndarray:
	if offspring_cnt is None:
		offspring_cnt = parent.shape[0]

	offsprings = np.array([])

	for _ in range(offspring_cnt):
		offspring = None

		if rec_type == 'intermediate':
			offspring = parent[np.random.choice(parent.shape[0], 2, replace=True)].mean(axis=0)
		elif rec_type == 'discrete':
			offspring = parent[np.random.choice(parent.shape[0], 2, replace=True)]
			choice = np.random.randint(2, size=offspring[0].size).reshape(offspring[0].shape).astype(bool)
			offspring = np.where(choice, *offspring)

		offsprings = np.append(offsprings, offspring)

	return offsprings.reshape(offspring_cnt, parent.shape[1])
