import numpy as np
import openbabel as ob

class Specie:
    """Core object to hold information and methods on a single specie"""

    def __init__(self, identifier=None, properties=None, charge=None):
        self.identifier = identifier
        if properties:
            self.properties = properties
        else:
            self.properties = {}
        self._id_properties = {} # dictionary of properties saved for ID string generation
        self.ac_matrix = None
        self.charge = charge

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
        s._id_properties['determinant'] = round(np.linalg.det(ac.matrix))
        s._id_properties['number_of_atoms'] = len(ac.matrix)
        # saves matrix as an attribute
        s.ac_matrix = ac
        return s

    def _get_id_str(self):
        s = ''
        for k in ['determinant', 'number_of_atoms']:
            if k in self._id_properties.keys():
                s += "_" + k + "_" + str(self._id_properties[k])
            else:
                s += k + "_NONE"
        return s


    def _get_charged_id_str(self):
        s = self._get_id_str()
        return s + "_charge_{}".format(int(self.charge) if not self.charge is None else "None")
        

    def has_charge(self):
        """Method to check if a specie has a defined charge"""
        return self.charge is not None

    def __eq__(self, x):
        # if not equal check properties
        conditions = []
        keys = set(list(self._id_properties.keys()) + list(x._id_properties.keys()))
        for k in keys:
            if k in self._id_properties.keys() and k in x._id_properties.keys():
                conditions.append(self._id_properties[k] == x._id_properties[k])
        return all(conditions)
