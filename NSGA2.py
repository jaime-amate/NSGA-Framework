import numpy as np
from NSGAInput import NSGAInput
from evolution import combine_population_real_representation
from non_dominated_sort import fast_non_dominated_sort


class NSGA2:
    def __init__(self, nsga_input: NSGAInput):
        """
        Initialize the NSGA2 algorithm with the given input parameters.

        Args:
            nsga_input (NSGAInput): An instance of NSGAInput containing the problem setup.
        """
        self.nsga_input = nsga_input

    def run(self, verbose: bool = False):
        """
        Run the NSGA-II algorithm.

        Args:
            verbose (bool): If True, prints progress information for each generation.
        """
        # Extract input parameters
        initial_population = self.nsga_input.initial_population
        target_functions = self.nsga_input.target_functions
        num_decision_variables = self.nsga_input.num_decision_variables
        objective_function = self.nsga_input.objective_function
        mutation_probability = self.nsga_input.mutation_probability

        # Evaluate the initial population
        evaluations = objective_function(initial_population)
        fronts, points_per_front = fast_non_dominated_sort(evaluations, target_functions)

        # Create offspring population (Q0)
        offspring_population_representation, offspring_evalutations = combine_population_real_representation(
            initial_population,
            num_decision_variables,
            fronts,
            np.empty(0),
            mutation_probability,
            objective_function,
            rank_binary_tournament,
            self.crossover_operator,
            self.mutation_operator,
            self.c_distribution_index,
            self.m_distribution_index,
        )

        # Iterate through generations
        for generation_index in np.arange(self.nsga_input.num_generations):
            if verbose:
                print('Generation:', generation_index)

            # Combine parent and offspring populations
            combined_population_representation = np.concatenate(
                (initial_population, self.offspring_population_representation), axis=0
            )
            self.combined_evaluations = np.concatenate(
                (self.evaluations, self.offspring_evalutations), axis=0
            )