from deap.tools import crossover
import copy
import random

def sbx_operator(ind1, ind2, distribution_index):

    individual1 = copy.deepcopy(ind1)
    individual12 = copy.deepcopy(ind2)

    return crossover.cxSimulatedBinaryBounded(individual1,individual12, distribution_index, 0.0, 1.0)

def set_crossover(ind1, ind2, n):

    #subset of features that have been selected by both parent individuals
    common_features = ind1 & ind2
    #remaining features in parent individual1 and indivudua2 once the common features are removed
    remaining_features = (ind1 | ind2) - common_features
    list_remaining_features = list(remaining_features)

    offspring1 = copy.deepcopy(common_features)
    offspring2 = copy.deepcopy(common_features)

    #remaining features are randomly distributed between offspring1 and offspring2
    while len(offspring1) != len(ind1):
        feature = random.choice(list_remaining_features)
        offspring1.add(feature)
        list_remaining_features.remove(feature)

    while len(offspring2) != len(ind2):
        feature = random.choice(list_remaining_features)
        offspring2.add(feature)
        list_remaining_features.remove(feature)

    return offspring1, offspring2
