import numpy as np
from non_dominated_sort import fast_non_dominated_sort
from selection import crowded_binary_tournament, rank_binary_tournament
from reference_points import generate_reference_points
from algorithms import NSGAII, NSGAIII
from dask.distributed import Client

class NSGAFramework:
    def __init__(self, population_size, p_mutation, num_objs, n, max_num_iterations, objective_function, generate_initial_population, crossover_operator, mutation_operator, target_functions, c_distribution_index, m_distribution_index, combine_method):
        self.population_size = population_size

        if population_size % 2 != 0:
            raise ValueError

        if num_objs != len(target_functions):
            raise ValueError

        self.combined_population_size = 2 * population_size

        self.n = n
        self.max_num_iterations = max_num_iterations

        self.p_mutation = p_mutation

        self.fronts = np.zeros(population_size)
        self.combined_fronts = np.zeros(population_size*2)
        self.num_objs = num_objs

        self.target_functions = target_functions

        self.objective_function = objective_function
        self.generate_initial_population = generate_initial_population
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.c_distribution_index = c_distribution_index
        self.m_distribution_index = m_distribution_index
        self.combine_method = combine_method

    def run_nsga_ii(self):

        self.algorithm = NSGAII()

        #random parent population (P0) is created
        self.population_representation = self.generate_initial_population(self.population_size)
        #evaluate
        self.evaluations = self.objective_function(self.population_representation)
        self.fronts, self.points_per_front = fast_non_dominated_sort(self.evaluations, self.target_functions)

        #create offspring population (Q0)
        self.offspring_population_representation, self.offspring_evalutations = self.combine_method(
            self.population_representation, self.n, self.fronts, np.empty(0),self.p_mutation, self.objective_function, rank_binary_tournament, self.crossover_operator, self.mutation_operator, self.c_distribution_index, self.m_distribution_index)


        for iteration_i in np.arange(self.max_num_iterations):

            print('Iteration: ', iteration_i)

            #Combine parent and offspring population
            self.combined_population_representation = np.concatenate(
                (self.population_representation, self.offspring_population_representation), axis=0)
            self.combined_evaluations = np.concatenate( (self.evaluations, self.offspring_evalutations), axis=0)

            #sort combined population according to nondomination
            self.fronts, self.points_per_front = fast_non_dominated_sort(self.combined_evaluations, self.target_functions)

            #choose the fisrt (N - |Pt+1|) elements of Fi
            self.population_representation,self.evaluations, self.crowding_assignment = self.algorithm.select_population(self.points_per_front, self.population_size, self.combined_population_representation, self.combined_evaluations)

            #use selection, crossover and mutation to create a new population
            # create offspring population (Q0)
            self.offspring_population_representation, self.offspring_evalutations = self.combine_method(
                self.population_representation, self.n, self.fronts, self.crowding_assignment, self.p_mutation, self.objective_function,
                crowded_binary_tournament, self.crossover_operator, self.mutation_operator, self.c_distribution_index, self.m_distribution_index)

            #increment the generation counter
        self.fronts, self.points_per_front = fast_non_dominated_sort(self.evaluations, self.target_functions)

    def run_nsga_iii(self):

        self.algorithm = NSGAIII()

        self.reference_points = generate_reference_points(self.num_objs, 12)
        self.num_reference_points = len(self.reference_points)

        self.z_min = np.full((self.num_objs), np.Inf)
        self.z_max = np.full((self.num_objs, self.num_objs), np.NINF)

        # random parent population (P0) is created
        self.population_representation = self.generate_initial_population(self.population_size)
        # evaluate
        self.evaluations = self.objective_function(self.population_representation)
        self.fronts, self.points_per_front = fast_non_dominated_sort(self.evaluations, self.target_functions)

        # create offspring population (Q0)
        self.offspring_population_representation, self.offspring_evalutations = self.combine_method(
            self.population_representation, self.n, self.fronts, np.empty(0), self.p_mutation, self.objective_function,
            rank_binary_tournament, self.crossover_operator, self.mutation_operator, self.c_distribution_index, self.m_distribution_index)

        for iteration_i in np.arange(self.max_num_iterations):
            print('Iteration: ', iteration_i)

            # Combine parent and offspring population
            self.combined_population_representation = np.concatenate(
                (self.population_representation, self.offspring_population_representation), axis=0)
            self.combined_evaluations = np.concatenate((self.evaluations, self.offspring_evalutations), axis=0)

            # sort combined population according to nondomination
            self.fronts, self.points_per_front = fast_non_dominated_sort(self.combined_evaluations,
                                                                         self.target_functions)

            # choose the fisrt (N - |Pt+1|) elements of Fi
            self.population_representation, self.evaluations = self.algorithm.select_population(self.points_per_front,
                                                                                              self.population_size,
                                                                                              self.combined_population_representation,
                                                                                              self.combined_evaluations,
                                                                                              self.target_functions,
                                                                                              self.z_min, self.z_max,
                                                                                              self.reference_points)

            # use selection, crossover and mutation to create a new population
            # create offspring population (Q0)
            self.offspring_population_representation, self.offspring_evalutations = self.combine_method(
                self.population_representation, self.n, self.fronts, np.empty(0), self.p_mutation,
                self.objective_function,
                rank_binary_tournament, self.crossover_operator, self.mutation_operator, self.c_distribution_index, self.m_distribution_index)

            # increment the generation counter
        self.fronts, self.points_per_front = fast_non_dominated_sort(self.evaluations, self.target_functions)
