import numpy as np

from sample import Sample


MUTATION_PROB = 0.3


def mutate_flip(sample: Sample, mutation_prob=MUTATION_PROB) -> Sample:
	for i in range(len(sample.x)):
		if np.random.rand() > MUTATION_PROB:
			sample.x[i] = 1 - sample.x[i]

	sample.fitness = sample.func(sample.x)
	return sample


def mutate_swap():
	pass


def mutate(sample: Sample, mutation_prob=MUTATION_PROB, mutation_type='flip'):
	type2function = {
		'flip': mutate_flip,
		'swap': mutate_swap,
	}

	return type2function[mutation_type](sample, mutation_prob)
