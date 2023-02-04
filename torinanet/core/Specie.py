import numpy as np
import openbabel as ob
from rdkit import Chem

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
        return self.ac_matrix.get_uid()

    def _get_charged_id_str(self):
        s = self._get_id_str()
        try:
            charge = int(self.charge)
        except ValueError:
            charge = "None"
        except TypeError:
            charge = "None"
        return s + "#{}".format(charge)

    def has_charge(self):
        """Method to check if a specie has a defined charge"""
        return self.charge is not None

    def to_rdmol(self):
        """convert specie to an RDKit molecule"""
        if self.ac_matrix is not None:
            return self.ac_matrix.to_rdmol()
        elif self.identifier is not None:
            return Chem.MolFromSmiles(self.identifier)
        else:
            return None

    def __eq__(self, x):
        if not type(x) is Specie:
            raise TypeError("Cannot compare Specie to {}".format(type(x)))
        # condition = ac matrices are equal and properties are equal
        return self.ac_matrix == x.ac_matrix and self.charge == x.charge
        
