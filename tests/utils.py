from typing import List, Iterable
import numpy as np
from torinanet.core import BinaryAcMatrix, Specie, Reaction

def join_matrices(mat1: np.ndarray, mat2: np.ndarray):
    """Method to make block matrix from two separate matrices. larger matrix is first."""
    l1 = len(mat1)
    l2 = len(mat2)
    if l2 > l1:
        return np.block([[mat2, np.zeros((l2, l1))], [np.zeros((l1, l2)), mat1]])
    else:
        return np.block([[mat1, np.zeros((l1, l2))], [np.zeros((l2, l1)), mat2]])

def good_ac(ac_mat: BinaryAcMatrix) -> bool:
    """Check if matrix is realistic"""
    max_bonds = {1: 1, 6: 4, 8: 2, 7: 3}
    for i, atom in enumerate(ac_mat.get_atoms()):
        if len(ac_mat.get_neighbors(i)) > max_bonds[atom]:
            return False
    return True


def reaction_generator(seed_species: List[Specie], max_changing_bonds: int) -> Iterable[Reaction]:
    """Random reaction generator using BinaryAcMatrix. this is supposed to emulate the behavior of an iterator but cheaper"""
    ajr = [BinaryAcMatrix.from_specie(s).to_specie() for s in seed_species]
    # cycling on species to make reactions
    while True:
        s1, s2 = np.random.choice(ajr, 2)
        mat = join_matrices(s1.ac_matrix.matrix, s2.ac_matrix.matrix)
        # making result matrix
        res = mat.copy()
        for _ in range(max_changing_bonds):
            # selecting indices
            i, j = np.random.randint(0, len(mat) - 1, size=2)
            # skipping illegal indices
            if i == j:
                continue
            # changing bonds on indices
            res[i, j] = 1 - res[i, j]
            res[j, i] = 1 - res[j, i]
        # making new reaction
        res = BinaryAcMatrix(res)
        if good_ac(res):
            reaction = Reaction.from_ac_matrices(BinaryAcMatrix(mat), res)
            if len(reaction.reactants) != 0 and len(reaction.products) != 0:
                # adding products to species
                ajr.extend(reaction.products)
                yield reaction

def specie_generator(seed_species: List[Specie], max_changing_bonds: int) -> Iterable[Reaction]:
    """Random reaction generator using BinaryAcMatrix. this is supposed to emulate the behavior of an iterator but cheaper"""
    ajr = [BinaryAcMatrix.from_specie(s).to_specie() for s in seed_species]
    seed = set([s.identifier for s in ajr])
    # cycling on species to make reactions
    while True:
        s1, s2 = np.random.choice(ajr, 2)
        mat = join_matrices(s1.ac_matrix.matrix, s2.ac_matrix.matrix)
        # making result matrix
        res = mat.copy()
        for _ in range(max_changing_bonds):
            # selecting indices
            i, j = np.random.randint(0, len(mat) - 1, size=2)
            # skipping illegal indices
            if i == j:
                continue
            # changing bonds on indices
            res[i, j] = 1 - res[i, j]
            res[j, i] = 1 - res[j, i]
        # making new reaction
        res = BinaryAcMatrix(res)
        if good_ac(res):
            reaction = Reaction.from_ac_matrices(BinaryAcMatrix(mat), res)
            # adding products to species
            for s in reaction.products:
                if not s.identifier in seed:
                    seed.add(s.identifier)
                    ajr.append(s)
                    yield s
