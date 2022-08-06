import unittest
from config import TORINA_NET_DIR, TORINA_X_DIR
import sys
sys.path.append(TORINA_NET_DIR)
sys.path.append(TORINA_X_DIR)
import torinanet as tn

class TestIterator (unittest.TestCase):

    def test_two_specie(self):
        s1 =tn.core.Specie(identifier="[CH]")
        s2 = tn.core.Specie(identifier="[H]")
        iterator = tn.iterate.Iterator(tn.core.RxnGraph())
        conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                                tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
        ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                        tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
        res = iterator.iterate_over_species(s1, s2, ac_filters, conv_filters).compute()
        for rxn in res.reactions:
            print("{} -> {}".format(" + ".join([s.ac_matrix.to_specie().identifier.strip() for s in rxn.reactants]),
                                    " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in rxn.products])))

if __name__ == "__main__":
    unittest.main()
