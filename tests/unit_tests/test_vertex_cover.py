import unittest
import os
from abc import ABC, abstractclassmethod
from config import torinanet as tn
from config import NETWORKS_DIR

class TestMvcFinder (unittest.TestCase, ABC):

    @abstractclassmethod
    def setUp(self) -> None:
        """Method to set up the analyzer used for each test, this is an abstract method to be implemented for each instance of MvcFinder to be tested.
        MUST define a self.analyzers attribute with dict of (name, MvcFinder) pairs. they all go through all tests"""
        pass

    def read_rxn_graph(self, name: str):
        """Method to read reaction graphs used in the tests"""
        if name == "base":
            return tn.core.RxnGraph.from_file(os.path.join(NETWORKS_DIR, "base.rxn"))
        else:
            raise ValueError("Unknown network name {}!".format(name))

    def base_case(self, analyzer):
        """Testing the base use-case of MvcFinder - finding MVC"""
        print("Testing base-use of MvcFinder")
        print("Finding MVC")
        rxn_graph = self.read_rxn_graph("base")
        mvc = analyzer.find_mvc(rxn_graph, verbose=1)
        print("Making sure a non-empty MVC is returned")
        self.assertIsNotNone(mvc)
        print("Asserting returning list of species")
        self.assertTrue(type(mvc) is list and all([type(s) is tn.core.Specie for s in mvc]))
        print("Found MVC with {} species".format(len(mvc)))
        print("Asserting reaction graph is unchanged")
        org_graph = self.read_rxn_graph("base")
        self.assertTrue(rxn_graph.is_equal(org_graph))
    
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
    unittest.main()
    # TestStochasticFinder().run()
    # TestGreedyFinder().run()