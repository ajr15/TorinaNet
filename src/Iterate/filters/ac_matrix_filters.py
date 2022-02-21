"""Module that contians possible ac matrix filters. ALL filters take only the ac_matrix as input."""
from abc import ABC, abstractclassmethod

class AcMatrixFilter (ABC):
    """Abstract AC matrix filter"""

    @abstractclassmethod
    def check(self, ac_matrix):
        """Method to check if the filter criterion holds for a given matrix"""
        pass


class MaxBondsPerAtom (AcMatrixFilter):
    """Filter that makes sure that atoms don't have too many bonds in the ac matrix"""

    def __init__(self, max_bonds_dict={1: 1, 6: 4, 7: 4, 8: 2, 9: 1}) -> None:
        self.max_bonds_dict = max_bonds_dict

    def check(self, ac_matrix):
        for i in range(len(ac_matrix)):
            if self.max_bonds_dict[ac_matrix.get_atom(i)] < len(ac_matrix.get_neighbors(i)):
                return False
        return True


class MaxAtomsOfElement (AcMatrixFilter):
    """Filter to limit the number of atoms of a specific element (for example carbon) in the ac_matrix.
    the filter takes """

    def __init__(self, max_atoms_of_element_dict: dict) -> None:
        self.max_atoms_of_element_dict = max_atoms_of_element_dict

    def check(self, ac_matrix):
        counter = {k: 0 for k in self.max_atoms_of_element_dict.keys()}
        for i in range(len(ac_matrix)):
            atom = ac_matrix.get_atom(i)
            if atom in counter:
                counter[atom] += 1
        return all([counter[k] <= self.max_atoms_of_element_dict[k] for k in self.max_atoms_of_element_dict.keys()])