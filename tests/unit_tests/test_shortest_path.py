import unittest
import torinanet as tn

class TestShortestPath (unittest.TestCase):

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

    def test_dijkstra(self):
        rxn_graph = self.make_reaction_graph()
        # running dijkstra's algorithm
        df = tn.analyze.algorithms.shortest_path_finders.dijkstra_shortest_path(rxn_graph)

class TestShortestPathFinder (unittest.TestCase):

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

    def setUp(self) -> None:
        self.rxn_graph = self.make_reaction_graph()
        self.finder = tn.analyze.algorithms.ShortestPathAnalyzer(self.rxn_graph)
        return super().setUp()

    def test_found_path(self):
        h2o = self.rxn_graph._read_specie_with_ac_matrix(tn.core.Specie("O"))
        ho2 = self.rxn_graph._read_specie_with_ac_matrix(tn.core.Specie("O[O]"))
        print("PATH OF H2O")
        print("distance =", self.finder.get_distance_from_source(h2o))
        print("path:")
        print("-----")
        for rxn in self.finder.get_path_to_source(h2o):
            print(rxn.pretty_string())
        print("PATH OF HO2")
        print("distance =", self.finder.get_distance_from_source(ho2))
        print("path:")
        print("-----")
        for rxn in self.finder.get_path_to_source(ho2):
            print(rxn.pretty_string())
        

if __name__ == "__main__":
    unittest.main()