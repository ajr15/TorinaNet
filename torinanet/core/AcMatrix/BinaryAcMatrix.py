from .AcMatrix import AcMatrix
from ..Specie import Specie
from .utils import join_molecules, guess_geometry, mm_geometry_optimization, sort_by_diagonal, obmol_to_molecule
import openbabel as ob
import networkx as nx
import numpy as np
from typing import Optional

class BinaryAcMatrix (AcMatrix):
    """Binary AC matrix object"""

    def __init__(self, matrix: Optional[np.ndarray]=None):
        self.matrix = matrix
        self.uid = None
    
    def get_atom(self, i: int):
        # the diagonal of the ac matrix is the list of atomic number
        if i > len(self.matrix):
            raise ValueError("too high index {}, number of atoms {}".format(i, len(self.matrix)))
        return self.matrix[i][i]

    def get_neighbors(self, i):
        ns = []
        for j, atom in enumerate(self.matrix[i]):
            if not j == i:
                if atom == 1:
                    ns.append(j)
        return ns

    def to_networkx_graph(self):
        """Abstract method to convert AC matrix object to an undirected NetworkX graph"""
        G = nx.Graph()
        # add atoms
        for i, z in enumerate(self.get_atoms()):
            # Adding information on atomic number to node
            G.add_node(i, Z=z)
        for i in range(len(self.matrix)):
            for j in range(i + 1, len(self.matrix)):
                if self.matrix[i][j] == 1:
                    G.add_edge(i, j)
        return G

    @staticmethod
    def from_networkx_graph(G: nx.Graph):
        """Method to read AC matrix from NetworkX object. Graph must contain data on atomic number as node properties."""
        mat = np.zeros((len(G.nodes), len(G.nodes)))
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        for u, v in G.edges:
            mat[node_to_idx[u], node_to_idx[v]] = 1
            mat[node_to_idx[v], node_to_idx[u]] = 1
        for i, n in enumerate(G.nodes(data=True)):
            mat[i, i] = n[-1]['Z']
        return BinaryAcMatrix(mat)

    @classmethod
    def from_specie(cls, specie: Specie, add_hydrogens: bool=True):
        """Abastract method to create AC matrix from a Specie"""
        if specie.ac_matrix is None:
            obmol = specie.parse_identifier()
            if add_hydrogens:
                obmol.AddHydrogens()
            return cls.from_obmol(obmol)
        else:
            return BinaryAcMatrix(specie.ac_matrix.matrix.copy())

    @staticmethod
    def from_obmol(obmol):
        ac = np.zeros((obmol.NumAtoms(), obmol.NumAtoms()))
        # writing off diagonal values - bond orders between atoms
        for i in range(obmol.NumBonds()):
            bond = obmol.GetBond(i)
            i = bond.GetBeginAtomIdx() - 1
            j = bond.GetEndAtomIdx() - 1
            ac[i, j] = 1
            ac[j, i] = 1
        # writing diagonal values - atom numbers
        for i in range(obmol.NumAtoms()):
            ac[i, i] = obmol.GetAtom(i + 1).GetAtomicNum()
        return BinaryAcMatrix(ac)

    def to_obmol(self) -> ob.OBMol:
        # using OpenBabel because of bond order perception method
        mol = ob.OBMol()
        # adding atoms
        for i in range(len(self.matrix)):
            atom = ob.OBAtom()
            atom.SetAtomicNum(int(self.get_atom(i)))
            mol.AddAtom(atom)
        # adding bonds
        for i in range(len(self.matrix)):
            for j in range(i + 1, len(self.matrix)):
                if self.matrix[i][j] == 1:
                    mol.AddBond(i + 1, j + 1, 1)
        return mol

    def to_specie(self) -> Specie:
        mol = self.to_obmol()
        # perceiving "correct" bond orders
        mol.PerceiveBondOrders()
        # getting SMILES string
        conv = ob.OBConversion()
        conv.SetOutFormat("smi")
        smiles = conv.WriteString(mol)
        # creating specie
        specie = Specie.from_ac_matrix(self)
        specie.identifier = smiles.strip()
        return specie

    def build_geometry(self, connected_molecules: Optional[dict]=None, force_field: str="UFF", n_steps: int=1000):
        # start from getting the guess geometry for the unbounded part of the molecule
        # sorting, to order by atoms and binding sites (integers > 0 and integers =< 0)
        sorted_mat = sort_by_diagonal(self.matrix)
        # finding the bounded part of the matrix
        bounded_idx = None
        for i in range(len(sorted_mat)):
            if sorted_mat[i, i] <= 0:
                bounded_idx = i
                break
        # reading as a specie
        unbound_mat = BinaryAcMatrix(sorted_mat[:bounded_idx, :bounded_idx])
        unbound_mol = unbound_mat.to_obmol()
        unbound_mol.PerceiveBondOrders()
        specie = obmol_to_molecule(unbound_mol)
        # making guess geometry for specie
        specie = guess_geometry(specie)
        # in case no connected molecues found, return specie
        if not bounded_idx:
            specie = mm_geometry_optimization(specie, force_field, n_steps)
            return specie
        # now connecting molecules (supports only one molecule for now)
        for i in range(bounded_idx, len(self.matrix)):
            molecule = connected_molecules[self.matrix[i, i]]["molecule"]
            binding_atom = connected_molecules[self.matrix[i, i]]["binding_atom"]
            specie_atom = self.get_neighbors(i)
            molecule = join_molecules(molecule, specie, binding_atom, specie_atom, 1.5)
        # optimizing geometry
        molecule = mm_geometry_optimization(molecule, force_field, n_steps)
        return molecule
