from deap.tools import mutation
import random
import numpy as np
import copy

def polynomial_mutation(ind, p_mutation, distribution_index):
    individual = copy.deepcopy(ind)
    return np.asarray(mutation.mutPolynomialBounded(individual,distribution_index,0.0,1.0,p_mutation))

def set_mutation(individual, p_mutation, n):

    M = set()
    # subset of features that will be mutated
    for i in individual:
        if random.uniform(0.0, 1.0) <= p_mutation:
            M.add(i)

    Ms = set()
    for m in M:
        if random.uniform(0.0, 1.0) <= 0.5:
            Ms.add(m)

    Mr = M - Ms

    # subset of new features that will substitute those belonging Ms
    Ns = set()
    while len(Ns) < len(Ms):
        rand_item = random.randint(0, n - 1)
        if not rand_item in individual:
            Ns.add(rand_item)
    try:
        Na = {random.choice(list( (individual-M) - Ns))}
    except IndexError:
        Na = set()
    mutated_individual = (individual - M) | Ns | Na

    if len(mutated_individual) == 0:
        return {(list(individual)[0]+1)%n}

    return mutated_individual
