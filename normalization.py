import numpy as np

def update_ideal_point(costs,z_min, j):

    new_ideal = np.min(costs[:, j])
    z_min[j] = min(new_ideal, z_min[j])

# Normalize objective functions of St (Structured points)
def normalize(combined_evaluations, structured_points, num_objs, target_functions, z_min, z_max):
    evaluations = combined_evaluations[structured_points]
    #print('Structured points costs: ', evaluations)

    for M in np.arange(num_objs):
        update_ideal_point(evaluations,z_min, M)
        perform_scalarizing(evaluations, z_max, M)

    translated_objectives = evaluations - z_min

    a = find_hyperplane_intercepts(evaluations, z_max)

    #print('Ideal point: ', z_min)
    #print('Extreme points: ', z_max)
    #print('Translated objetives: ', translated_objectives)
    #print('Intercepts: ', a)

    normalized_evaluations = np.zeros((len(structured_points), num_objs))

    #before speed up
    '''
    for i in np.arange(len(structured_points)):
        normalized_evaluations[i] = translated_objectives[i] / a
    '''

    normalized_evaluations = translated_objectives / a

    #print('Normalized costs: ', normalized_evaluations)

    return normalized_evaluations

def perform_scalarizing(evaluations, z_max, j):
    ind = find_extreme_point_objective(evaluations, j)

    if evaluations[ind][j] > z_max[j][j]:
        z_max[j] = evaluations[ind]

def find_extreme_point_objective(evaluations, j):
    #Finds the invidivuals with extreme values for each objective function.
    return np.argmax(evaluations[:, j])

def find_hyperplane_intercepts(evaluations, z_max):

    num_objs = evaluations.shape[1]
    intercepts = np.zeros(num_objs)

    #check if duplicated individuals
    if np.all(evaluations == np.unique(evaluations, axis=0)):
        for j in range(num_objs):
            intercepts[j] = z_max[j][j]
    else:
        b = np.ones(num_objs)
        try:
            x = np.linalg.solve(z_max,b)
        except np.linalg.LinAlgError:
            for j in range(num_objs):
                intercepts[j] = z_max[j][j]
        else:
            intercepts = 1/x

    return intercepts
