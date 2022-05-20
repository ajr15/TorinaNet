from torinax.base import Molecule, Bond
from torinax.utils.openbabel import obmol_to_molecule, molecule_to_obmol
import openbabel as ob
from typing import Optional
import numpy as np

def make_translation_vector(molecule: Molecule, binding_atom: int, size: float) -> np.ndarray:
    active = molecule.atoms[binding_atom]
    neighbors = molecule.get_neighbors(binding_atom)
    v1 = active.coordinates - molecule.atoms[neighbors[0]].coordinates
    v2 = active.coordinates - molecule.atoms[neighbors[1]].coordinates
    res = np.array([v1[1] * v2[2] - v1[2] * v2[1],
                        v1[2] * v2[0] - v1[0] * v2[2],
                        v1[0] * v2[1] - v1[1] * v2[0]])
    return size * res / np.linalg.norm(res)

def join_molecules(mol1: Molecule,
                    mol2: Molecule, 
                    connected_atom1: int, 
                    connected_atom2: int, 
                    translation_vector_size: Optional[float]=None) -> Molecule:
    # making both bounded atoms as (0, 0, 0)
    mol1.move_by_vector(np.negative(mol1.atoms[connected_atom1].coordinates))
    mol2.move_by_vector(np.negative(mol2.atoms[connected_atom2].coordinates))
    if not translation_vector_size is None:
        translation_vec = make_translation_vector(mol1, connected_atom1, translation_vector_size)
    else:
        translation_vec = np.zeros(3)
    mol2.move_by_vector(translation_vec)
    mol1.join(mol2)
    mol1.add_bond(Bond(connected_atom1, connected_atom2 + len(mol1.atoms) - len(mol2.atoms), 1))
    return mol1

class OpenbabelBuildError (Exception):
    pass

def guess_geometry(molecule: Molecule) -> Molecule:
    obmol = molecule_to_obmol(molecule)
    builder = ob.OBBuilder()
    if not builder.Build(obmol):
        conv = ob.OBConversion()
        raise OpenbabelBuildError("Failed building the smiles: ".format(conv.WriteString(obmol)))
    return obmol_to_molecule(obmol)

class OpenbabelFfError (Exception):
    pass

def mm_geometry_optimization(molecule: Molecule, force_field: str="UFF", nsteps: int=1000) -> Molecule:
    obmol = molecule_to_obmol(molecule)
    # optimization
    OBFF = ob.OBForceField.FindForceField(force_field)
    suc = OBFF.Setup(obmol)
    if not suc == True:
        raise OpenbabelFfError("Could not set up force field for molecule")
    OBFF.ConjugateGradients(nsteps)
    OBFF.GetCoordinates(obmol)
    return obmol_to_molecule(obmol)

def sort_by_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Method to sort a matrix by its diagonal values"""
    m, n = matrix.shape
    # finds the ith minimum diagonal value
    for i in range(m):
        sm = matrix[i, i]
        pos = i
        # find maximum in unsorted matrix
        for j in range(i + 1, n):
            if sm < matrix[j, j]:
                sm = matrix[j, j]
                pos = j
        # now switching minimum (i) with maximum (pos)
        # switching rows
        matrix[[i, pos]] = matrix[[pos, i]]
        # switching columns
        matrix[:, [i, pos]] = matrix[:, [pos, i]]
    return matrix
