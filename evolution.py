import numpy as np
import random

def combine_population_real_representation(population_representation, n, fronts, crowding_assignment, p_mutation, objective_function, binary_tournament, crossover_operator, mutation_operator, c_distribution_index, m_distribution_index):
    
    population_size = len(population_representation)
    offspring_population_representation = np.zeros((population_size, n))
    
    i = 0
    while i < population_size:
        #select two random parents
        random_individuals = random.sample(range(0,population_size),2)
        #extract first best parent
        ind1 = binary_tournament(random_individuals[0], random_individuals[1], fronts, crowding_assignment)
        #select two random parents
        random_individuals = random.sample(range(0,population_size),2)
        #extract second best parent
        ind2 = binary_tournament(random_individuals[0], random_individuals[1], fronts, crowding_assignment)

        offspring1, offspring2 = crossover_operator(population_representation[ind1], population_representation[ind2], c_distribution_index)

        offspring_population_representation[i] = mutation_operator(offspring1, p_mutation, m_distribution_index)
        offspring_population_representation[i+1] = mutation_operator(offspring2, p_mutation, m_distribution_index)

        i+=2

    offspring_evalutations = objective_function(offspring_population_representation)

    return offspring_population_representation, offspring_evalutations


def combine_population_set_representation(population_representation, n, fronts, crowding_assignment, p_mutation, objective_function,
                       binary_tournament, crossover_operator, mutation_operator, c_distribution_index,
                       m_distribution_index):

    population_size = len(population_representation)
    offspring_population_representation = np.empty(population_size, dtype=object)

    i = 0
    while i < population_size:
        # select two random parents
        random_individuals = random.sample(range(0, population_size), 2)
        # extract first best parent
        ind1 = binary_tournament(random_individuals[0], random_individuals[1], fronts, crowding_assignment)
        # select two random parents
        random_individuals = random.sample(range(0, population_size), 2)
        # extract second best parent
        ind2 = binary_tournament(random_individuals[0], random_individuals[1], fronts, crowding_assignment)

        offspring1, offspring2 = crossover_operator(population_representation[ind1], population_representation[ind2],n)

        offspring_population_representation[i] = mutation_operator(offspring1, p_mutation, n)
        offspring_population_representation[i + 1] = mutation_operator(offspring2, p_mutation, n)

        i += 2

    offspring_evalutations = objective_function(offspring_population_representation)

    return offspring_population_representation, offspring_evalutations
