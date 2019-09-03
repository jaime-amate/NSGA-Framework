import numpy as np
from operator import itemgetter
import random
import dask
from numba import jit

#@jit(nogil=True)
def perpendicular_distance(direction, point):
    k = np.dot(direction, point) / np.sum(np.power(direction, 2))
    d = np.sum(np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point), 2))
    
    return np.sqrt(d)

def associate(structured_points, normalized_evaluations, reference_points, next_generation):
    reference_points_assignment = np.zeros(len(structured_points))
    reference_points_perpencidular_distances = np.zeros(len(structured_points))
    association_counts_next_generation = np.zeros(len(reference_points))
    association_counts_structured_points = np.zeros(len(reference_points))

    for i in range(len(structured_points)):
        #compute distances between the point i in St and all the reference points
        reference_points_distances = np.fromiter(
            [perpendicular_distance(normalized_evaluations[i], reference_point) for reference_point in
             reference_points], float)

        #select the closest reference point
        nearest_reference_point = np.argmin(reference_points_distances)
        #assign the distance of the closest point
        reference_points_perpencidular_distances[i] = reference_points_distances[nearest_reference_point]
        #assign the reference point associated to the point of St
        reference_points_assignment[i] = nearest_reference_point
        #increment the association count of the reference point (niche count)
        association_counts_structured_points[nearest_reference_point] += 1
        if structured_points[i] in next_generation:
            association_counts_next_generation[nearest_reference_point] += 1

    #print(reference_points)
    #print("Reference point association: ", reference_points_assignment)
    #print("Association counts to Structuted Points (St)", association_counts_structured_points)
    #print("Association counts to Next generation (Pt+1)", association_counts_next_generation)
    #print("Distances to reference points: ", reference_points_perpencidular_distances)

    return reference_points_assignment, association_counts_structured_points, association_counts_next_generation, reference_points_perpencidular_distances

#@jit(nogil=True)
def compute_closest_reference_point(point, reference_points):
    #compute distances between the point i in St and all the reference points
    reference_points_distances = [perpendicular_distance(point, reference_point) for reference_point in reference_points]
    #select the closest reference point
    nearest_reference_point = np.argmin(reference_points_distances)

    return nearest_reference_point, reference_points_distances[nearest_reference_point]


def associate_dask(structured_points, normalized_evaluations, reference_points, next_generation):
    association_counts_next_generation = np.zeros(len(reference_points))
    association_counts_structured_points = np.zeros(len(reference_points))

    delayed_reference_points = dask.delayed(reference_points)

    delayed_results = [
        dask.delayed(compute_closest_reference_point)(normalized_evaluations[i], delayed_reference_points) for i in
        range(len(structured_points))]

    results = dask.compute(*delayed_results)
    results = np.asarray(results)

    reference_points_assignment = results[:, 0].astype('int')
    reference_points_perpencidular_distances = results[:, 1]

    unique_counts, counts_elements = np.unique(reference_points_assignment, return_counts=True)
    association_counts_structured_points[unique_counts] = counts_elements

    unique_counts, counts_elements = np.unique(
        reference_points_assignment[np.where(structured_points == next_generation)], return_counts=True)
    association_counts_next_generation[unique_counts] = counts_elements

    return reference_points_assignment, association_counts_structured_points, association_counts_next_generation, reference_points_perpencidular_distances

def associate_to_niche(structured_points, association_counts_next_generation, reference_points_assignment, reference_points_perpencidular_distances, last_front_points, last_front_index,num_K):
    selected_last_front_points = set()

    while len(selected_last_front_points) < num_K:
        #indentify the reference point having minimum niche count
        min_reference_point = np.argsort(association_counts_next_generation)[0]
        #print("Min reference point: ", min_reference_point)
        #print("Association counts to Structured Points (St)", association_counts_structured_points)
        #print("Association counts to Next generation (Pt+1)", association_counts_next_generation)

        if association_counts_next_generation[min_reference_point] == 0:
            '''
            points_associated_minimum_niche = []
            for i in range(last_front_index, len(structured_points)):
                if reference_points_assignment[i] == min_reference_point:
                    points_associated_minimum_niche.append(i)
            '''
            points_associated_minimum_niche = [i for i in range(last_front_index, len(structured_points)) if reference_points_assignment[i] == min_reference_point]

            if len(points_associated_minimum_niche) == 0:
                association_counts_next_generation[min_reference_point] = np.Inf
            else:
                perpendicular_distances = [(point, reference_points_perpencidular_distances[point]) for point in
                                           points_associated_minimum_niche]
                min_perpendicular_distance_point = sorted(perpendicular_distances, key=itemgetter(1))[0]

                selected_last_front_points.add(structured_points[min_perpendicular_distance_point[0]])
                association_counts_next_generation[min_reference_point] += 1
        else:
            random_last_front_member = random.randrange(last_front_points.size)

            #print(random_last_front_member, last_front_index + random_last_front_member,
                  #structured_points[last_front_index + random_last_front_member])

            if reference_points_assignment[
                last_front_index + random_last_front_member] == min_reference_point:
                selected_last_front_points.add(structured_points[last_front_index + random_last_front_member])

            association_counts_next_generation[min_reference_point] += 1

        #print('Selected points from last front: ', selected_last_front_points)

    return list(selected_last_front_points)