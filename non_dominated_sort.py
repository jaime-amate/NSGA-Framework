import numpy as np


def domination_operator(u, v):
    num_objs = u.shape[0]

    for j in np.arange(num_objs):
        if u[j] > v[j]:
            return False

    return not np.array_equal(u, v)

def fast_non_dominated_sort(evaluations, target_functions):
    num_objs = len(target_functions)
    normalized_evaluations = np.copy(evaluations)
    population_size = evaluations.shape[0]

    dominated_solutions = np.empty(population_size, dtype=object)
    domination_counts = np.zeros(population_size)
    fronts = np.full(population_size, np.inf)
    points_per_front = []
    first_front_points = []

    for j in np.arange(num_objs):
        if target_functions[j] == 1:
            normalized_evaluations[:, j] *= -1

    for p in np.arange(population_size):

        dominated_solutions[p] = set()

        for q in np.arange(population_size):

            if domination_operator(normalized_evaluations[p], normalized_evaluations[q]):  # if p dominates q
                dominated_solutions[p].add(q)  # add q to the set of solutions dominated by p
            elif domination_operator(normalized_evaluations[q], normalized_evaluations[p]):
                domination_counts[p] += 1  # increment the domination counter of p

        if domination_counts[p] == 0:  # p belongs to the first front
            fronts[p] = 0
            first_front_points.append(p)

    points_per_front.append(first_front_points)

    front_counter = 0  # initialize the front counter
    while not len(points_per_front[front_counter]) == 0:
        Q = set()  # used to store the members of the next front

        for p in points_per_front[front_counter]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:  # q belongs to the next front
                    fronts[q] = front_counter + 1
                    Q.add(q)
        front_counter += 1
        points_per_front.append(list(Q))

    return fronts, points_per_front[:-1]