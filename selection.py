
def rank_operator(ind1, ind2, fronts):
    return fronts[ind1] < fronts[ind2]

def crowded_operator(ind1, ind2, fronts, crowding_assignment):
    return fronts[ind1] < fronts[ind2] or (fronts[ind1] == fronts[ind2] and crowding_assignment[ind1] > crowding_assignment[ind2])

def crowded_binary_tournament(ind1, ind2, fronts, crowding_assignment):

    if crowded_operator(ind1, ind2, fronts, crowding_assignment):
        return ind1
    else:
        return ind2

def rank_binary_tournament(ind1, ind2, fronts, crowding_assignment):

    if rank_operator(ind1, ind2, fronts):
        return ind1
    else:
        return ind2