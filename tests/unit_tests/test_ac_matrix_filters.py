import torinanet as tn
import openbabel as ob
import unittest

def ac_from_smiles(smiles: str):
    conv = ob.OBConversion()
    mol = ob.OBMol()
    conv.SetInFormat("smi")
    conv.ReadString(mol, smiles)
    mol.AddHydrogens()
    return tn.core.BinaryAcMatrix.from_obmol(mol)

class TestMaxRingNumber (unittest.TestCase):

    SMILES_AND_RINGS = [
        ("[CH3]", 0),
        ("C1CC1", 1),
        ("C1C2CCC12", 2),
        ("C1CC1C1CC1", 2),
        ("c1cc(ccc1)C1CC1C1CC1", 3),
        ("N1ON1", 1)
    ]

    def test_filter(self):
        print("Testing the MaxRingNumber filter")
        filters = [tn.iterate.ac_matrix_filters.MaxRingNumber(0),
                    tn.iterate.ac_matrix_filters.MaxRingNumber(1),
                    tn.iterate.ac_matrix_filters.MaxRingNumber(2)]
        for smiles, n_rings in self.SMILES_AND_RINGS:
            print(smiles)
            ac = ac_from_smiles(smiles)
            for f in filters:
                res = f.check(ac)
                print("max rings={}, pass={}".format(f.max_rings, res))
                self.assertTrue((n_rings <= f.max_rings) == res)

class TestMinRingSize (unittest.TestCase):

    SMILES_AND_PROP = [
        ("[CH3]", 0),
        ("C1CC1", 3),
        ("C1C2CCC12", 4),
        ("C1CCC(C1)C1CCCC1", 5),
        ("c1cc(ccc1)C1CC1C1CC1", 3),
        ("N1ON1", 3)
    ]

    def test_filter(self):
        print("Testing the MinRingSize filter")
        filters = [tn.iterate.ac_matrix_filters.MinRingSize(3),
                    tn.iterate.ac_matrix_filters.MinRingSize(5),
                    tn.iterate.ac_matrix_filters.MinRingSize(6)]
        for smiles, prop in self.SMILES_AND_PROP:
            print(smiles)
            ac = ac_from_smiles(smiles)
            for f in filters:
                res = f.check(ac)
                if prop > 0:
                    ref = (prop >= f.min_atoms)
                else:
                    ref = True
                print("min_size={}, pass={}".format(f.min_atoms, res))
                self.assertTrue(ref == res)

class TestHeteroRingFilter (unittest.TestCase):

    SMILES_AND_PROP = [
        ("[CH3]", [0, 0]),
        ("C1CC1", [0, 3]),
        ("C=1C=C/C(C=1)=C1/CC=CN1", [1, 5]),
        ("C=1C=C/C([O+]=1)=C1\CC=CN1", [1, 5]),
        ("C1NNCO1", [3, 5]),
        ("N1ON1", [3, 3]),
        ("[o+]1ccccc1", [1, 6])
    ]

    def test_filter(self):
        print("Testing the HeteroRingFilter filter")
        filters = [tn.iterate.ac_matrix_filters.HeteroRingFilter(1, 5),
                    tn.iterate.ac_matrix_filters.HeteroRingFilter(1, 6),
                    tn.iterate.ac_matrix_filters.HeteroRingFilter(0, 0)]
        for smiles, prop in self.SMILES_AND_PROP:
            print(smiles)
            max_hetero, min_ring = prop
            ac = ac_from_smiles(smiles)
            for f in filters:
                res = f.check(ac)
                ref = (max_hetero <= f.max_heteroatoms) and (min_ring >= f.min_ring_atoms) or (max_hetero == 0)
                print("min_size={}, max_hetero={}, pass={}".format(f.min_ring_atoms, f.max_heteroatoms, res))
                self.assertTrue(ref == res)

class TestMaxRingsPerAtom (unittest.TestCase):

    SMILES_AND_PROP = [
        ("[CH3]", 0),
        ("C1CC1", 1),
        ("C1C2CCC12", 2),
        ("C1CCC2CCC3CCCC132", 3),
        ("N1ON1", 1)
    ]

    def test_filter(self):
        print("Testing the MaxRingsPerAtom filter")
        filters = [tn.iterate.ac_matrix_filters.MaxRingsPerAtom(0),
                    tn.iterate.ac_matrix_filters.MaxRingsPerAtom(1),
                    tn.iterate.ac_matrix_filters.MaxRingsPerAtom(2)]
        for smiles, prop in self.SMILES_AND_PROP:
            print(smiles)
            ac = ac_from_smiles(smiles)
            for f in filters:
                res = f.check(ac)
                ref = (prop <= f.max_rings)
                print("max_rings={}, pass={}".format(f.max_rings, res))
                self.assertTrue(ref == res)

class TestBondOrderRingFilter (unittest.TestCase):

    SMILES_AND_PROP = [
        ("[CH3]", [0, 0]),
        ("C1CC1", [1, 3]),
        ("C1C=C1", [2, 3]),
        ("C1=CCC1C1C=CCC1", [2, 4]),
        ("C1#CCC1C1CC#CC1C1CC1", [3, 4])
    ]

    def test_filter(self):
        print("Testing the BondOrderRingFilter filter")
        filters = [tn.iterate.ac_matrix_filters.BondOrderRingFilter({2: 3}),
                    tn.iterate.ac_matrix_filters.BondOrderRingFilter({2: 4}),
                    tn.iterate.ac_matrix_filters.BondOrderRingFilter({2: 3, 3: 4})]
        for smiles, prop in self.SMILES_AND_PROP:
            print(smiles)
            bo, ring_size = prop
            ac = ac_from_smiles(smiles)
            for f in filters:
                res = f.check(ac)
                if bo in f.bond_order_dict:
                    ref = ring_size > f.bond_order_dict[bo]
                else:
                    ref = True
                print("dict={}, pass={}".format(f.bond_order_dict, res))
                self.assertTrue(ref == res)

if __name__ == "__main__":
    unittest.main()
