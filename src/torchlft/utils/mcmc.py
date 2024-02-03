from itertools import pairwise

def metropolis_acceptance_from_indices(indices: list[int]) -> float:
    #return len(set(indices)) / len(indices) # out by one?
    n_acc = indices[0] + sum([j - i for (i, j) in pairwise(indices)])
    r_acc = n_acc / len(indices)
    return r_acc
