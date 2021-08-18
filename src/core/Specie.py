import numpy as np
import openbabel as ob

class Specie:
    """Core object to hold information and methods on a single specie"""

    def __init__(self, identifier=None, properties={}):
        self.identifier = identifier
        self.properties = properties
        self.ac_matrix = None

    def parse_identifier(self):
        """Abstract static method to parse the identifier to a Specie object"""
        conv = ob.OBConversion()
        conv.SetInFormat("smi")
        mol = ob.OBMol()
        conv.ReadString(mol, self.identifier)
        return mol

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

    def _get_id_str(self):
        s = ''
        for k in ['determinant', 'number_of_atoms']:
            if k in self.properties.keys():
                s += "_" + k + "_" + str(self.properties[k])
            else:
                s += k + "_NONE"
        return s

    def __eq__(self, x):
        # if not equal check properties
        conditions = []
        keys = set(list(self.properties.keys()) + list(x.properties.keys()))
        for k in keys:
            if k in self.properties.keys() and k in x.properties.keys():
                conditions.append(self.properties[k] == x.properties[k])
        return all(conditions)
