import numpy as np
from crowding import crowding_distance_assignment
from normalization import normalize
from niching import associate, associate_dask, associate_to_niche


class NSGAII:

    def generate_structured_points(self, points_per_front, population_size, evaluations):
        structured_points = []
        front_counter = 0
        crowding_per_front = []
        crowding_assignment = np.empty(0)

        while len(structured_points) < population_size:

            for point in points_per_front[front_counter]:
                structured_points.append(point)

            crowding_per_front.append( crowding_distance_assignment (evaluations[points_per_front[front_counter]]))
            crowding_assignment = np.append(crowding_assignment, crowding_per_front[front_counter])

            front_counter += 1

        last_front = front_counter - 1
        last_front_points = points_per_front[last_front]

        return np.asarray(structured_points, dtype=int), last_front, np.asarray(last_front_points, dtype=int), crowding_assignment, crowding_per_front

    def select_population(self, points_per_front, population_size, combined_population_representation, combined_evaluations):

        structured_points, last_front, last_front_points, crowding_assignment, crowding_per_front = self.generate_structured_points(points_per_front, population_size, combined_evaluations)

        if len(structured_points) == population_size:
            # print('Next generation: ', structured_points)
            population_representation = combined_population_representation[structured_points]
            evaluations = combined_evaluations[structured_points]

            return population_representation, evaluations, crowding_assignment

        else:
            num_elements = len(structured_points) - len(last_front_points)
            next_generation = structured_points[:num_elements]
            num_K = population_size - len(next_generation)

            #order by crowding distance of solutions of the last from (Fl)
            ordered_crowded = np.argsort(crowding_per_front[last_front]*-1)
            selected_points = last_front_points[ordered_crowded[:num_K]]
            selected_crowds = crowding_per_front[last_front][ordered_crowded[:num_K]]

            population_representation = np.concatenate((combined_population_representation[next_generation],
                                                        combined_population_representation[selected_points]), axis=0)
            evaluations = np.concatenate((combined_evaluations[next_generation], combined_evaluations[selected_points]),
                                         axis=0)

            crowding_assignment = np.concatenate( (crowding_assignment[:len(next_generation)], selected_crowds),axis=0 )

            return population_representation, evaluations, crowding_assignment

class NSGAIII:


    def generate_structured_points(self, points_per_front, population_size):
        structured_points = []
        front_counter = 0
        while len(structured_points) < population_size:

            for point in points_per_front[front_counter]:
                structured_points.append(point)
            front_counter += 1

        last_front = front_counter - 1
        last_front_points = points_per_front[last_front]

        return np.asarray(structured_points, dtype=int), np.asarray(last_front_points, dtype=int)

    def select_population(self, points_per_front, population_size, combined_population_representation, combined_evaluations, target_functions, z_min, z_max, reference_points):

        structured_points, last_front_points = self.generate_structured_points(points_per_front, population_size)
        num_objs = len(target_functions)

        if len(structured_points) == population_size:
            #print('Next generation: ', structured_points)
            population_representation = combined_population_representation[structured_points]
            evaluations = combined_evaluations[structured_points]

            return population_representation, evaluations

        else:
            #print('Normalization needed.')
            #print('Structured points: ', structured_points)
            #print('Last front points: ', last_front_points)

            num_elements = len(structured_points) - len(last_front_points)
            next_generation = structured_points[:num_elements]
            #print('Next generation: ', next_generation)

            last_front_index = len(next_generation)
            num_K = population_size - len(next_generation)

            #print('Num points to be chosen from last front: ', num_K)

            normalized_evaluations = normalize(combined_evaluations, structured_points, num_objs, target_functions,
                                               z_min, z_max)

            reference_points_assignment, association_counts_structured_points, association_counts_next_generation, reference_points_perpencidular_distances = \
                associate_dask(structured_points, normalized_evaluations, reference_points, next_generation)

            selected_points = \
                associate_to_niche(structured_points, association_counts_next_generation,
                        reference_points_assignment, reference_points_perpencidular_distances, last_front_points,
                        last_front_index, num_K);

            population_representation = np.concatenate((combined_population_representation[next_generation],
                                                        combined_population_representation[selected_points]), axis=0)
            evaluations = np.concatenate((combined_evaluations[next_generation], combined_evaluations[selected_points]),
                                         axis=0)

            return population_representation, evaluations