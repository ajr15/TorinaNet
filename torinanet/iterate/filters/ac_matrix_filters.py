"""Module that contians possible ac matrix filters. ALL filters take only the ac_matrix as input."""
from abc import ABC, abstractclassmethod
from typing import Optional

from click import Option

class AcMatrixFilter (ABC):
    """Abstract AC matrix filter"""

    @abstractclassmethod
    def check(self, ac_matrix):
        """Method to check if the filter criterion holds for a given matrix"""
        pass


class MaxBondsPerAtom (AcMatrixFilter):
    """Filter that makes sure that atoms don't have too many bonds in the ac matrix.
    ARGS:
        - max_bonds_dict (dict): dictionary with maximum number of bonds (values) for each atom (keys, atom numbers)
        - dative_bonds_with_dummy_atoms (bool): weather to count bonds with dummy atoms as regular bonds for the filter
        - dative_atoms (list): list of atoms to not count bonds with dummy atoms for"""

    def __init__(self, max_bonds_dict: Optional[dict]=None,
                 dative_bonds_with_dummy_atoms: bool=True, dative_atoms: Optional[list]=None) -> None:
        self.skip_dummies = dative_bonds_with_dummy_atoms
        # setting max bonds per atom
        if not max_bonds_dict is None:
            self.max_bonds_dict = max_bonds_dict
        else:
            self.max_bonds_dict = {1: 1, 6: 4, 7: 4, 8: 2, 9: 1}
        # setting list of dative atoms
        if not dative_atoms is None:
            self.dative_atoms = dative_atoms
        else:
            self.dative_atoms = [1, 8, 7]

    def bond_counter(self, ac_matrix, atom_idx):
        neigbors = ac_matrix.get_neighbors(atom_idx)
        if self.skip_dummies:
            # if we want to regard bonds with dummy atoms as dative
            # we cannot count them as regular bonds
            if ac_matrix.get_atom(atom_idx) in self.dative_atoms:
                # if atom has dative capabilities, dont count dummies as neighbors
                return sum([1 if ac_matrix.get_atom(n) > 0 else 0 for n in neigbors])
            else:
                return len(neigbors)

    def check(self, ac_matrix):
        for i in range(len(ac_matrix)):
            if self.max_bonds_dict[ac_matrix.get_atom(i)] < self.bond_counter(ac_matrix, i):
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