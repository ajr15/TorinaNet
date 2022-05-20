from config import TORINA_NET_DIR, TORINA_X_DIR
import sys
sys.path.append(TORINA_NET_DIR)
sys.path.append(TORINA_X_DIR)
import os
import unittest
import torinanet as tn


def make_rxn_strings(rxn_graph: tn.core.RxnGraph):
    strs = set()
    for reaction in rxn_graph.reactions:
        s = " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.reactants]) + \
                " -> " + \
                " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.products])
        strs.add(s)
    return s


class TestRxnGraph (unittest.TestCase):

    def setUp(self) -> None:
        """Sets up a default graph for testing - decomposition of H2O molecule"""
        # setting up graph
        print("Setting up reaction graph...")
        print("Using enumerated network of H2O decomposition with 2 macro-iterations")
        rxn_graph = tn.core.RxnGraph()
        rxn_graph.set_source_species([tn.core.Specie("O")], force=True)
        stop_cond = tn.iterate.stop_conditions.MaxIterNumber(1)
        ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom()]
        conversion_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(2), 
                                tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
        iterator = tn.iterate.iterators.Iterator(rxn_graph)
        self.h2o_graph = iterator.enumerate_reactions(conversion_filters, ac_filters, stop_cond, verbose=0)
        print("DONE SETUP")

    def test_save_graph(self):
        """Testing the method for saving graph data"""
        print("Testing the rxn_graph.save method")
        original_rxn_strs = make_rxn_strings(self.h2o_graph)
        print("saving graph")
        graph_path = "./test.rxn"
        self.h2o_graph.save(graph_path)
        self.assertTrue(os.path.isfile(graph_path))
        print("testing saved graph")
        read_graph = tn.core.RxnGraph.from_file(graph_path)
        rxn_strs = make_rxn_strings(read_graph)
        self.assertTrue(rxn_strs == original_rxn_strs)
        original_source = set([self.h2o_graph.make_unique_id(s) for s in self.h2o_graph.source_species])
        read_source = set([self.h2o_graph.make_unique_id(s) for s in read_graph.source_species])
        self.assertTrue(original_source == read_source)


if __name__ == '__main__':
    unittest.main()