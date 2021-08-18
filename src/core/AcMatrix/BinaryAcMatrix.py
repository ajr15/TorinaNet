from .AcMatrix import AcMatrix
from ..Specie import Specie
import openbabel as ob
import networkx as nx
import numpy as np

class BinaryAcMatrix (AcMatrix):
    """Binary AC matrix object"""

    def __init__(self, matrix):
        self.matrix = matrix
    
    def get_atom(self, i):
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
                if self.matrix[i][j] ==1:
                    G.add_edge(i, j)
        return G

    @staticmethod
    def from_networkx_graph(G):
        """Method to read AC matrix from NetworkX object. Graph must contain data on atomic number as node properties."""
        mat = np.zeros((len(G.nodes), len(G.nodes)))
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        for u, v in G.edges:
            mat[node_to_idx[u], node_to_idx[v]] = 1
            mat[node_to_idx[v], node_to_idx[u]] = 1
        for i, n in enumerate(G.nodes(data=True)):
            mat[i, i] = n[-1]['Z']
        return BinaryAcMatrix(mat)

    @staticmethod
    def from_specie(specie, add_hydrogens=True):
        """Abastract method to create AC matrix from a Specie"""
        if specie.ac_matrix is None:
            obmol = specie.parse_identifier()
            if add_hydrogens:
                rdmol = obmol.AddHydrogens()
            ac = np.zeros((obmol.NumAtoms(), obmol.NumAtoms()))
            # writing off diagonal values - bond orders between atoms
            for i in range(obmol.NumBonds()):
                bond = obmol.GetBond(i)
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                ac[i, j] = 1
                ac[j, i] = 1
            # writing diagonal values - atom numbers
            for i in range(rdmol.NumAtoms()):
                ac[i, i] = rdmol.GetAtom(i).GetAtomicNum()
            return BinaryAcMatrix(ac)
        else:
            return BinaryAcMatrix(specie.ac_matrix.matrix)

    def to_specie(self):
        # using OpenBabel because of bond order perception method
        mol = ob.OBMol()
        # adding atoms
        for i in range(len(self.matrix)):
            atom = ob.OBAtom()
            atom.SetAtomicNum(self.get_atom(i))
            mol.AddAtom(atom)
        # adding bonds
        for i in range(len(self.matrix)):
            for j in range(i + 1, len(self.matrix)):
                mol.AddBond(i, j, 1)
        # perceiving "correct" bond orders
        mol.PercieveBondOrder()
        # getting SMILES string
        conv = ob.OBConversion()
        conv.SetOutFormat("smi")
        smiles = conv.WriteString(mol)
        # creating specie
        specie = Specie.from_ac_matrix(self)
        specie.identifier = smiles
        return specie