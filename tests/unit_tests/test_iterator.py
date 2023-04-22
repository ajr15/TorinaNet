import unittest
import torinanet as tn

class TestIterator (unittest.TestCase):

    # @unittest.skip("")
    def test_two_specie(self):
        g = tn.core.RxnGraph()
        s1 = g._read_specie_with_ac_matrix(tn.core.Specie("C"))
        s2 = g._read_specie_with_ac_matrix(tn.core.Specie("[H]"))
        iterator = tn.iterate.Iterator(tn.core.RxnGraph(), dask_scheduler="processes")
        conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                                tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
        ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                        tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
        res = iterator.iterate_over_species(s1, s2, ac_filters, conv_filters).compute()
        print("Iteration over {} + {}".format(s1.identifier, s2.identifier))
        for rxn in res.reactions:
            print("{} -> {}".format(" + ".join([s.ac_matrix.to_specie().identifier.strip() for s in rxn.reactants]),
                                    " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in rxn.products])))

    # @unittest.skip("")
    def test_single_specie(self):
        g = tn.core.RxnGraph()
        s1 = g._read_specie_with_ac_matrix(tn.core.Specie("C"))
        iterator = tn.iterate.Iterator(tn.core.RxnGraph(), dask_scheduler="processes")
        conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                                tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
        ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                        tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
        res = iterator.iterate_over_a_specie(s1, ac_filters, conv_filters).compute()
        print("Iteration over", s1.identifier)
        for rxn in res.reactions:
            print("{} -> {}".format(" + ".join([s.ac_matrix.to_specie().identifier.strip() for s in rxn.reactants]),
                                    " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in rxn.products])))

    def test_iterate(self):
        g = tn.core.RxnGraph()
        g.set_source_species([
            tn.core.Specie("O")
        ], force=True)
        iterator = tn.iterate.Iterator(g, dask_scheduler="processes")
        conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                                tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
        ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                        tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
        stop_condition = tn.iterate.stop_conditions.MaxIterNumber(2)
        res = iterator.enumerate_reactions(conv_filters, ac_filters, stopping_condition=stop_condition)
        print("Enumeration over CH4 network")
        for rxn in res.reactions:
            print("{} -> {}".format(" + ".join([s.ac_matrix.to_specie().identifier.strip() for s in rxn.reactants]),
                                    " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in rxn.products])))        


if __name__ == "__main__":
    unittest.main()
