"""Module that contians possible ac matrix filters. ALL filters take only the ac_matrix as input."""
from abc import ABC, abstractclassmethod
from typing import Optional
import openbabel as ob


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
        # if we count all neigbors as bonds, return length of neighbor list
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

    def check_mat(self, ac_matrix) -> bool:
        counter = {k: 0 for k in self.max_atoms_of_element_dict.keys()}
        for i in range(len(ac_matrix)):
            atom = ac_matrix.get_atom(i)
            if atom in counter:
                counter[atom] += 1
        return all([counter[k] <= self.max_atoms_of_element_dict[k] for k in self.max_atoms_of_element_dict.keys()])

    def check(self, ac_matrix) -> bool:
        for ac in ac_matrix.get_compoenents():
            if not self.check_mat(ac_matrix):
                return False
        return True

class MaxComponents (AcMatrixFilter):

    """Filter that prevents AC matrices with more than N components to be admitted. this puts a limit on the number of products in a reaction"""

    def __init__(self, n: int) -> None:
        self.n = n

    def check(self, ac_matrix) -> bool:
        return len(ac_matrix.get_compoenents()) <= self.n

class MaxRingNumber (AcMatrixFilter):
    """Filter out molecules with more than max_rings rings"""

    def __init__(self, max_rings: int) -> None:
        super().__init__()
        self.max_rings = max_rings

    def check(self, ac_matrix):
        """Method to check if the filter criterion holds for a given matrix"""
        obmol = ac_matrix.to_obmol()
        # finding minimal number of smallest rings (SSSR)
        obmol.FindSSSR()
        return len(obmol.GetSSSR()) <= self.max_rings
    
class MinRingSize (AcMatrixFilter):
    """Filter out molecules with less than 'min_atoms' atoms"""

    def __init__(self, min_atoms: int) -> None:
        super().__init__()
        self.min_atoms = min_atoms

    def check(self, ac_matrix):
        """Method to check if the filter criterion holds for a given matrix"""
        obmol = ac_matrix.to_obmol()
        # finding minimal number of smallest rings (SSSR)
        obmol.FindSSSR()
        return all([ring.Size() >= self.min_atoms for ring in obmol.GetSSSR()])

    
class HeteroRingFilter (AcMatrixFilter):

    """Filter to rule out species with unlikely heteroatoms given their composition. filters rings with more than 'max_heteroatoms' or with less than this amount and rings with less atoms than 'min_ring_atoms'"""

    def __init__(self, max_heteroatoms: int, min_ring_atoms: int) -> None:
        super().__init__()
        self.max_heteroatoms = max_heteroatoms
        self.min_ring_atoms = min_ring_atoms

    def check(self, ac_matrix):
        obmol = ac_matrix.to_obmol()
        # finding minimal number of smallest rings (SSSR)
        obmol.FindSSSR()
        rings = obmol.GetSSSR()
        # counting number of non-carbon (heteroatoms) atoms in each ring
        hetero_counter = [0 for _ in range(len(rings))]
        for atom in ob.OBMolAtomIter(obmol):
            if atom.GetAtomicNum() != 6:
                for i, ring in enumerate(rings):
                    if ring.IsMember(atom):
                        hetero_counter[i] += 1
        # retuning whether all rings have OK number of heteroatoms wrt their size
        return all([(hetero <= self.max_heteroatoms and ring.Size() >= self.min_ring_atoms) or hetero == 0 for hetero, ring in zip(hetero_counter, rings) if hetero > 0])
    
class MaxRingsPerAtom (AcMatrixFilter):

    """filters molecules with more than 'max_rings' per atom"""

    def __init__(self, max_rings) -> None:
        super().__init__()
        self.max_rings = max_rings

    def check(self, ac_matrix):
        obmol = ac_matrix.to_obmol()
        # finding minimal number of smallest rings (SSSR)
        obmol.FindSSSR()
        rings = obmol.GetSSSR()
        for atom in ob.OBMolAtomIter(obmol):
            counter = 0
            for ring in rings:
                if ring.IsMember(atom):
                    counter += 1
            if counter > self.max_rings:
                return False
        return True

class BondOrderRingFilter (AcMatrixFilter):

    """Filters molecules with rings that consist small rings with high bond orders"""

    def __init__(self, bond_order_dict: dict) -> None:
        super().__init__()
        self.bond_order_dict = bond_order_dict
        
    def check(self, ac_matrix):
        obmol = ac_matrix.to_obmol()
        obmol.PerceiveBondOrders()
        # finding minimal number of smallest rings (SSSR)
        obmol.FindSSSR()
        rings = obmol.GetSSSR()
        for bond in ob.OBMolBondIter(obmol):
            for ring in rings:
                # if bond is in ring, try to estimate its order by the valence of the atoms
                if ring.IsMember(bond):
                    begin_diff = bond.GetBeginAtom().GetImplicitValence() - bond.GetBeginAtom().GetValence()
                    end_diff = bond.GetEndAtom().GetImplicitValence() - bond.GetEndAtom().GetValence()
                    bo = min(1 + begin_diff, 1 + end_diff)
                    if bo in self.bond_order_dict and ring.Size() <= self.bond_order_dict[bo]:
                        return False
        return True
