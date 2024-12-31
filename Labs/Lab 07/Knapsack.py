import numpy as np
import random

items = [(3,5),(2,4),(1,8),(6,8),(8,10)]
weight_limit = 10
population_size = 10
generations = 20
mutation_rate = 0.2

def initialize_population(size, num_items):
    population = []
    for _ in range(size):
        solution = np.random.randint(2, size=num_items)
        population.append(solution)
    return population


def calculate_fitness(solution):
    total_weight = sum(solution[i] * items[i][0] for i in range(len(solution)))
    total_value = sum(solution[i] * items[i][1] for i in range(len(solution)))
    if total_weight <= weight_limit:
        return total_value
    else:
        return 0

def select_parent(population, fitnesses):
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    selected_index = np.random.choice(len(population), p=selection_probs)
    return population[selected_index]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def mutate(solution):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = 1 - solution[i]
    return solution

def genetic_algorithm_knapsack():
    num_items = len(items)
    population = initialize_population(population_size, num_items)

    for generation in range(generations):
        fitnesses = [calculate_fitness(solution) for solution in population]

        new_population = []
        for _ in range(population_size // 2):
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        population = new_population

    best_solution = max(population, key=calculate_fitness)
    best_value = calculate_fitness(best_solution)
    return best_solution, best_value

best_solution, best_value = genetic_algorithm_knapsack()
print("Best Solution (Items included):", best_solution)
print("Best Value:", best_value)
