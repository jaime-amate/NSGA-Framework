import numpy as np

class NSGAInput:
    def __init__(
        self,
        num_objectives: int,
        num_decision_variables: int,
        mutation_probability: float,
        initial_population: np.ndarray,
        objective_function: callable,
        target_functions: list, 
        population_size: int = 100, 
        num_generations: int = 250, 
        crossover_distribution_index: float = 20.0,
        mutation_distribution_index: float = 20.0
    ):
        if population_size % 2 != 0:
            raise ValueError('Population size must be even.')
        if num_objectives != len(target_functions):
            raise ValueError('Number of objectives must be equal to the number of target functions.')

        self.num_objectives = num_objectives
        self.num_decision_variables = num_decision_variables
        self.mutation_probability = mutation_probability
        self.objective_function = objective_function
        self.population_size = population_size
        self.combined_population_size = 2 * population_size
        self.num_generations = num_generations
        self.crossover_distribution_index = crossover_distribution_index
        self.mutation_distribution_index = mutation_distribution_index
        self.initial_population = initial_population

