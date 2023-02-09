import unittest
import os
from abc import ABC, abstractclassmethod
import torinanet as tn

class TestMvcFinder (unittest.TestCase):

    def setUp(self) -> None:
        """Method to set up the analyzer used for each test, this is an abstract method to be implemented for each instance of MvcFinder to be tested.
        MUST define a self.analyzers attribute with dict of (name, MvcFinder) pairs. they all go through all tests"""
        pass

    def make_reaction_graph(self):
        print("Making graph of H2O")
        g = tn.core.RxnGraph()
        g.set_source_species([
            tn.core.Specie("O")
        ], force=True)
        iterator = tn.iterate.Iterator(g)
        conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                                tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
        ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                        tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
        stop_condition = tn.iterate.stop_conditions.MaxIterNumber(2)
        return iterator.enumerate_reactions(conv_filters, ac_filters, stopping_condition=stop_condition, verbose=0)


    def base_case(self, analyzer):
        """Testing the base use-case of MvcFinder - finding MVC"""
        print("Testing base-use of MvcFinder")
        print("Finding MVC")
        rxn_graph = self.make_reaction_graph()
        mvc = analyzer.find_mvc(rxn_graph, verbose=1)
        print("Making sure a non-empty MVC is returned")
        self.assertIsNotNone(mvc)
        print("Asserting returning list of species")
        self.assertTrue(type(mvc) is list and all([type(s) is tn.core.Specie for s in mvc]))
        print("Found MVC with {} species".format(len(mvc)))
        print("Asserting reaction graph is unchanged")
        org_graph = self.make_reaction_graph()
        self.assertTrue(rxn_graph == org_graph)
    
    def test_base_case(self):
        for name, analyzer in self.analyzers.items():
            print("Testing for", name, "MVC finder")
            self.base_case(analyzer)

class TestStochasticFinder (TestMvcFinder):

    """Testing the stochastic MVC finding object"""

    def setUp(self) -> None:
        self.analyzers = {
            "degree-useAllSpecies": tn.analyze.algorithms.vertex_cover.StochasticMvcFinder("degree", 300, False),
            "degree-useOnlyProducts": tn.analyze.algorithms.vertex_cover.StochasticMvcFinder("degree", 300, True),
        }

class TestGreedyFinder (TestMvcFinder):

    """Testing the stochastic MVC finding object"""

    def setUp(self) -> None:
        self.analyzers = {
            "greedy-degree": tn.analyze.algorithms.vertex_cover.GreedyMvcFinder("degree")
        }

if __name__ == '__main__':
    runner = unittest.TestSuite([TestGreedyFinder("test_base_case"), TestStochasticFinder("test_base_case")])
    runner.run(unittest.TestResult())
