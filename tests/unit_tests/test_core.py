import numpy as np
import unittest
from itertools import permutations
from rdkit.Chem import rdFingerprintGenerator
import torinanet as tn

def scrambled_species(ac: np.ndarray):
    """Method to take ac matrix and scramble it to get all possible variations (as species"""
    edge_list = [(i, j) for i, j in zip(*np.where(ac == 1)) if i != j]
    atoms = [ac[i, i] for i in range(len(ac))]
    # going over all permutations of the atoms in the AC matrix
    for combo in permutations(range(len(ac)), len(ac)):
        # making a re-mapped edge-list
        ajr = [(combo[i], combo[j]) for i, j in edge_list]
        # building ac from new edge list
        res = np.zeros_like(ac)
        # setting atoms on diagonal
        for i in range(len(ac)):
            res[combo[i], combo[i]] = atoms[i]
        # setting scambled bonds
        for i, j in ajr:
            res[i, j] = 1
        yield tn.core.BinaryAcMatrix(res).to_specie()


class TestSpecie (unittest.TestCase):

    def test_equality(self):
        """Test if the specie equality is working properly"""
        # collections of variations on HC#N molecule
        hcns = list(scrambled_species(np.array([
                    [1, 0, 1],
                    [0, 7, 1], 
                    [1, 1, 8]
                ])))
        # variations on the HN#C molecule
        hncs = list(scrambled_species(np.array([
                    [1, 1, 0],
                    [1, 7, 1], 
                    [0, 1, 8]
                ])))
        # making sure all both sets are equal
        for s1 in hcns:
            for s2 in hcns:
                self.assertTrue(s1 == s2)
        for s1 in hncs:
            for s2 in hncs:
                self.assertTrue(s1 == s2)
        # making sure the two sets are different
        for s1 in hncs:
            for s2 in hcns:
                self.assertFalse(s1 == s2)
        # testing "in" method
        self.assertTrue(hcns[0] in hcns[1:])
        self.assertTrue(hncs[0] in hncs[1:])
        self.assertTrue(all([s not in hncs for s in hcns]))
        self.assertTrue(all([s not in hcns for s in hncs]))


class TestReaction (unittest.TestCase):

    @staticmethod
    def rxn_string(rxn):
        return "{} = {}".format(" + ".join([s.identifier for s in rxn.reactants]), " + ".join([s.identifier for s in rxn.products]))


    def test_from_ac_matrices(self):
        """Test the 'from_ac_matrices' method for reaction reading"""
        ac1 = np.array([
            [7, 1, 1],
            [1, 1, 0],
            [0, 1, 1]
        ])
        ac2 = np.array([
            [7, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        rxn = tn.core.Reaction.from_ac_matrices(tn.core.BinaryAcMatrix(ac1), tn.core.BinaryAcMatrix(ac2))
        print(self.rxn_string(rxn))


    def test_equality(self):
        pass

@unittest.skip("Because")
class TestBinaryMatrix (unittest.TestCase):

    # def build_smiles(self, smiles: str):
    #     specie = tn.core.Specie(smiles)
    #     ac = tn.core.BinaryAcMatrix.from_specie(specie)
    #     molecule = ac.build_geometry(force_field="MMFF94")
    #     molecule.save_to_file("{}.xyz".format(smiles))
    #     # cleaning up
    #     os.remove("{}.xyz".format(smiles))
    
    # def test_build_specie(self):
    #     print("building C1CCCC1")
    #     self.build_smiles("C1CCCC1")
    #     print("building c1occccc1")
    #     self.build_smiles("c1occccc1")
    #     print("building CCC(CC)O")
    #     self.build_smiles("CCC(CC)O")
    #     print("building CO2")
    #     self.build_smiles("O=C=O")

    def test_to_rdkit(self):
        specie = tn.core.Specie("N")
        ac = tn.core.BinaryAcMatrix.from_specie(specie)
        rdmol = ac.to_rdmol()
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=128, numBitsPerFeature=4)
        fpgen.GetFingerprint(rdmol)


if __name__ == "__main__":
    unittest.main()