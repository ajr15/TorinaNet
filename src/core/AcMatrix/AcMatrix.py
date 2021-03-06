from abc import ABC, abstractstaticmethod, abstractclassmethod
import networkx as nx
import numpy as np

class AcMatrix (ABC):

    @abstractclassmethod
    def __init__(self, matrix):
        self.matrix = matrix

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

    def __len__(self):
        return len(self.matrix)

    def copy(self):
        return self.__init__(self.matrix.copy())
