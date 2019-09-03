import numpy as np
from NSGAFramework import NSGAFramework
from feature_selection import feature_selection
from crossover import set_crossover
from mutation import set_mutation
from plotting import  plot_evaluations, plot_reference_points, plot_3d_evaluations
from evolution import combine_population_set_representation
import time

problem = feature_selection('datasets/sonar.all-data.csv')
population_size = 300
p_mutation = 0.001
max_num_iterations = 100
c_distribution_index = 0.0
m_distribution_index = 0.0

nsga = NSGAFramework(
    population_size,
    p_mutation,
    problem.num_objs,
    problem.n,
    max_num_iterations,
    problem.objective_function,
    problem.generate_initial_population,
    set_crossover,
    set_mutation,
    problem.target_functions,
    c_distribution_index,
    m_distribution_index,
    combine_population_set_representation
)

begin = time.time()
nsga.run_nsga_ii()
end = time.time()

pareto_front_points = nsga.points_per_front[0]
pareto_front_evaluations = nsga.evaluations[pareto_front_points]
pareto_front = nsga.fronts[pareto_front_points]
evaluations = nsga.evaluations

print('Computation time: ', end-begin)

print(pareto_front_evaluations)
print(nsga.population_representation[pareto_front_points])
plot_evaluations(pareto_front_evaluations, problem)