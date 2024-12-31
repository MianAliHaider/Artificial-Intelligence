import numpy as np
import random

distance_matrix = np.array([
    [0, 10, 15, 20, 10],
    [10, 0, 35, 25, 15],
    [15, 35, 0, 30, 20],
    [20, 25, 30, 0, 25],
    [10, 15, 20, 25, 0]
])


population_size = 25
generations = 50
mutation_rate = 0.1

def initialize_population(size, num_cities):
    population = []
    for _ in range(size):
        tour = random.sample(range(num_cities), num_cities)
        population.append(tour)
    return population

def calculate_fitness(tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance_matrix[tour[i], tour[i+1]]
    total_distance += distance_matrix[tour[-1], tour[0]]
    return 1 / total_distance

def select_parent(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    selected_index = np.random.choice(len(population), p=selection_probs)
    return population[selected_index]

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]

    for city in parent2:
        if city not in child:
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = city
                    break
    return child

def mutate(tour):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def genetic_algorithm_tsp():
    num_cities = len(distance_matrix)
    population = initialize_population(population_size, num_cities)

    for generation in range(generations):
        fitnesses = [calculate_fitness(tour) for tour in population]

        new_population = []
        for _ in range(population_size // 2):
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        population = new_population

    best_tour = max(population, key=calculate_fitness)
    best_distance = 1 / calculate_fitness(best_tour)
    return best_tour, best_distance

best_tour, best_distance = genetic_algorithm_tsp()
print("Tour:", best_tour)
print("Distance:", best_distance)
