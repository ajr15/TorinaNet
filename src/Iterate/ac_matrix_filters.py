"""Module that contians possible ac matrix filters. ALL filters take only the ac_matrix as input."""

# TODO: define a better dictionary
max_changing_bonds_dict = {
    1: 1,
    6: 4,
    7: 4,
    8: 2,
    9: 1
}

def max_bonds_per_atom(ac_matrix):
    """Filter that makes sure that atoms don't have too many bonds in the ac matrix"""
    for i in range(len(ac_matrix)):
        if max_changing_bonds_dict[ac_matrix.get_atom(i)] < len(ac_matrix.get_neighbors(i)):
            return False
    return True