from NSGAFramework import NSGAFramework
from benchmarks import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, DTLZ1, DTLZ2, DTLZ3
from crossover import sbx_operator
from mutation import polynomial_mutation
from plotting import  plot_evaluations, plot_reference_points, plot_3d_reference_points, plot_3d_evaluations, plot_benchmark_evaluations
from evolution import combine_population_real_representation

if __name__ == '__main__':

    problem = DTLZ1(3)
    population_size = 100
    p_mutation = 1.0 / problem.n
    max_num_iterations = 250
    c_distribution_index = 20.0
    m_distribution_index = 20.0

    nsga = NSGAFramework(
        population_size,
        p_mutation,
        problem.num_objs,
        problem.n,
        max_num_iterations,
        problem.objective_function,
        problem.generate_initial_population,
        sbx_operator,
        polynomial_mutation,
        problem.target_functions,
        c_distribution_index,
        m_distribution_index,
        combine_population_real_representation
    )

    pareto_front_points = nsga.points_per_front[0]
    pareto_front_evaluations = nsga.evaluations[pareto_front_points]
    pareto_front = nsga.fronts[pareto_front_points]
    evaluations = nsga.evaluations

    print('Computation time: ', end-begin)
    print(evaluations)
    #plot_benchmark_evaluations(evaluations, problem)
    plot_3d_evaluations(evaluations, problem)
    #plot_reference_points(nsga.reference_points)
    #plot_3d_reference_points(nsga.reference_points)