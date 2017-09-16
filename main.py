import numpy as np
import math
import random
import matplotlib.pyplot as plt

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


def fitness(individual, adj_mat):
    dist = 0
    # Trajectory distance
    for i in range(len(individual)-1):
        src = individual[i]
        dst = individual[i+1]
        dist += adj_mat[src, dst]
    # Return to origin
    dist += adj_mat[i, 0]

    # Fitness improves as the path shortens
    return -dist

def pop_fitness(population, adj_mat):
    n_individuals, _ = population.shape
    fit = np.zeros(n_individuals)

    for i in range(n_individuals):
        fit[i] = fitness(population[i], adj_mat)

    return fit


def mutation(individual):
    print(individual)
    # Swapping
    positions = np.random.randint(0, len(individual), 2)
    print(positions)
    aux = individual[positions[0]]
    individual[positions[0]] = individual[positions[1]]
    individual[positions[1]] = aux


def crossover(population, best_individual_i, prob_mut=5e-2):
    n_individuals, n_cities = population.shape
    n_crs = population.shape[1]
    parent_1_inheritance_size = random.randint(1, math.ceil(n_crs / 2))
    parent_2_inheritance_size = n_cities - parent_1_inheritance_size

    best_individual = population[best_individual_i]

    # Reproducing best individual with everyone
    for i in range(n_individuals):
        # Best individual survives to reproduce
        if i == best_individual_i:
            continue

        new_individual = np.full(n_crs, -1)
        individual = population[i]

        starting_point = random.randint(0, n_cities - parent_1_inheritance_size)

        best_individual_contribution = best_individual[starting_point:starting_point + parent_1_inheritance_size]
        new_individual[starting_point:starting_point + parent_1_inheritance_size] = best_individual_contribution

        individual_contribution = [na for na in individual if na not in best_individual_contribution]

        parent_2_starting_point = starting_point + parent_1_inheritance_size
        for j in range(parent_2_inheritance_size):
            insert_pos = (parent_2_starting_point + j) % n_cities
            new_individual[insert_pos] = individual_contribution[j]

        prob = np.random.rand()
        # Event happened
        if prob > prob_mut:
            mutation(new_individual)

        population[i] = new_individual


def run(n_cities=100, n_generations=10e3):
    adj_mat = get_adj_mat(n_cities)
    population = get_population(10, n_cities)

    i = 0
    results = []
    while i < n_generations:
        fit = pop_fitness(population, adj_mat)
        best_fit = fit.max()
        results.append(best_fit)
        best_individual_i = fit.argmax()
        crossover(population, best_individual_i)

        i += 1

        print(i, results)

    x = range(int(n_generations))
    y = results
    plt.plot(x, y)
    plt.show()

run()