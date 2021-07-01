import numpy as np
from rdkit import Chem

class Specie:
    """Core object to hold information and methods on a single specie"""

    def __init__(self, identifier=None, properties={}):
        self.identifier = identifier
        self.properties = properties
        self.ac_matrix = None

    def parse_identifier(self):
        """Abstract static method to parse the identifier to a Specie object"""
        return Chem.MolFromSmiles(self.identifier, sanitize=True)

    @staticmethod
    def from_ac_matrix(ac):
        """Method to read a Specie from an AC matrix"""
        s = Specie(None, {})
        # calculates important (and quick) properties of matrix
        s.properties['determinant'] = np.linalg.det(ac.matrix)
        s.properties['number_of_atoms'] = len(ac.matrix)
        # saves matrix as an attribute
        s.ac_matrix = ac
        return s

    def __eq__(self, x):
        # if not equal check properties
        conditions = []
        keys = set(list(self.properties.keys()) + list(x.properties.keys()))
        for k in keys:
            if k in self.properties.keys() and k in x.properties.keys():
                conditions.append(self.properties[k] == x.properties[k])
        return all(conditions)
