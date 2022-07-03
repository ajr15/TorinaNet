from config import TORINA_NET_DIR, TORINA_X_DIR
import sys
sys.path.append(TORINA_NET_DIR)
sys.path.append(TORINA_X_DIR)
import unittest
import torinanet as tn


class TestBinaryMatrix (unittest.TestCase):

    def build_smiles(self, smiles: str):
        specie = tn.core.Specie(smiles)
        ac = tn.core.BinaryAcMatrix.from_specie(specie)
        molecule = ac.build_geometry(force_field="MMFF94")
        molecule.save_to_file("{}.xyz".format(smiles))
    
    def test_build_specie(self):
        print("building C1CCCC1")
        self.build_smiles("C1CCCC1")
        print("building c1occccc1")
        self.build_smiles("c1occccc1")
        print("building CCC(CC)O")
        self.build_smiles("CCC(CC)O")
        print("building CO2")
        self.build_smiles("O=C=O")


if __name__ == "__main__":
    unittest.main()