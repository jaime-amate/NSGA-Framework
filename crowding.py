import numpy as np

def crowding_distance_assignment(evaluations):
    population_size = evaluations.shape[0]
    num_objs = evaluations.shape[1]
    crowding_assignment = np.zeros(population_size)

    for m in np.arange(num_objs):

        # f_max = np.max(evaluations[:,m])
        # f_min = np.min(evaluations[:,m])
        sorted_evaluations = np.argsort(evaluations[:, m])  # sort using each objective value

        crowding_assignment[
            sorted_evaluations[0]] = np.inf  # so that the boundary points are always selected for all other points
        crowding_assignment[sorted_evaluations[-1]] = np.inf

        f_min = evaluations[sorted_evaluations[0]][m]
        f_max = evaluations[sorted_evaluations[-1]][m]

        for i in np.arange(1, population_size - 1):
            crowding_assignment[sorted_evaluations[i]] = crowding_assignment[sorted_evaluations[i]] + (
                        evaluations[sorted_evaluations[i + 1]][m] - evaluations[sorted_evaluations[i - 1]][m]) / (
                                                                     f_max - f_min)


    return crowding_assignment