import numpy as np
import random

def get_adj_mat(n_cities):
    adj_mat = np.random.rand(n_cities, n_cities)
    adj_mat *= 100
    adj_mat = np.matrix(adj_mat, dtype=np.int64)
    return adj_mat

def get_population(n_individuals, crs_size):
    population = np.zeros((n_individuals, crs_size), dtype=np.int64)

    for i in range(n_individuals):
        individual = list(range(crs_size))
        np.random.shuffle(individual)
        population[i] = individual

    return population

def run(n_cities=15):
    adj_mat = get_adj_mat(n_cities)
    population = get_population(n_cities, n_cities)

run()