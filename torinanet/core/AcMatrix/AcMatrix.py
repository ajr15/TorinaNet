from abc import ABC, abstractstaticmethod, abstractclassmethod
from typing import Optional
import networkx as nx
import numpy as np

class AcMatrix (ABC):

    @abstractclassmethod
    def __init__(self, matrix):
        self.matrix = matrix
        self.uid = None


    def get_uid(self) -> str:
        if self.uid is not None:
            return self.uid
        else:
            eigens = np.linalg.eigvalsh(self.matrix)
            poly = np.poly(eigens)
            self.uid = "".join(["{:0>5d}".format(abs(int(round(x, 0)))) for x in poly[1:]])
            return self.uid


    @abstractclassmethod
    def get_atom(self, i):
        """Abstract method to get the atomic number of the atom in the i-th index in AC matrix.
        ARGS:
            - i (int): index of the desired atom
        RETURNS:
            (int) atomic number of atom"""
        raise NotImplementedError

    @abstractclassmethod
    def get_neighbors(self, i):
        """Abstract method to get the neighbors (as list of atom indecis) of atom in the i-th index in the AC matrix.
        ARGS:
            - i (int): index of the desired atom
        RETURNS:
            (list) list of atom indecis of the neighboring atoms"""
        raise NotImplementedError

    @abstractclassmethod
    def to_networkx_graph(self):
        """Abstract method to convert AC matrix object to an undirected NetworkX graph"""
        raise NotImplementedError

    @abstractstaticmethod
    def from_networkx_graph(G):
        """Abstract method to read AC matrix from NetworkX object"""
        raise NotImplementedError

    @abstractstaticmethod
    def from_specie(specie):
        """Abastract method to create AC matrix from a Specie"""
        raise NotImplementedError

    @abstractclassmethod
    def to_specie(self):
        """Abastract method to convert an AC matrix to a Specie"""
        raise NotImplementedError

    def get_compoenents(self):
        """Method to decompose an AC matrix to its components"""
        G = self.to_networkx_graph()
        graph_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        return [self.from_networkx_graph(c) for c in graph_components]
    
    def get_atoms(self):
        atoms = []
        for i in range(len(self)):
            atoms.append(self.get_atom(i))
        return atoms

    def _to_str(self) -> str:
        """Method to write the ac matrix as a string (for internal use)"""
        string = ""
        for row in self.matrix:
            for x in row:
                string += str(x) + ","
            string += ";"
        return string

    def _from_str(self, string: str) -> None:
        """Method to read ac matrix from string (for internal use)"""
        string_matrix = [s.split(",") for s in string.split(";")]
        l = len(string_matrix) - 1
        matrix = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                if len(string_matrix[i][j]) > 0:
                    matrix[i, j] = float(string_matrix[i][j])
        self.__init__(matrix)

    def __len__(self):
        return len(self.matrix)

    def copy(self):
        return self.__init__(self.matrix.copy())

    def build_geometry(self, connected_molecules: Optional[dict]=None):
        """Method to build a geometry for the AC matrix, given a dictionary with dummy atom information.
        ARGS:
            - connected_molecules (dict): dictionary with dummy atom masses (keys) 
                                        and molecule structures (TorinaX species) + connecting atom index in molecule.
                                        for example:
                                            {-1: {catalyst_molecule1, 2}
        RETURNS:
            (TorinaX.Specie) TorinaX structure or molecule"""
        raise NotImplementedError("Cannot build geometry for the required AcMatrix object")

    def __eq__(self, other):
        det1 = np.linalg.det(self.matrix)
        det2 = np.linalg.det(other.matrix)
        l1 = len(self.matrix)
        l2 = len(other.matrix)
        return det1 == det2 and l1 == l2
