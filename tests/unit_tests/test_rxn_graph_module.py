from time import time
import os
import unittest
from typing import List
from torinanet.core import RxnGraph, Reaction, Specie
from tests.utils import reaction_generator


class TestRxnGraph (unittest.TestCase):

    @staticmethod
    def make_rxn_string(reaction: Reaction) -> str:
        s = " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.reactants]) + \
            " -> " + \
            " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.products])
        return s

    @classmethod
    def make_graph(cls, n_reactions: int, seed_species: List[Specie], max_changing_bonds: int=4):
        print("Making reaction graph")
        rxn_graph = RxnGraph()
        ntrails = 0
        for rxn in reaction_generator(seed_species, max_changing_bonds):
            ntrails += 1
            print("Adding {} n_reactions = {}, n_species = {}".format(cls.make_rxn_string(rxn), rxn_graph.get_n_reactions(), rxn_graph.get_n_species()))
            rxn_graph.add_reaction(rxn)
            if rxn_graph.get_n_reactions() >= n_reactions:
                return rxn_graph, ntrails
    
    @unittest.skip("")
    def test_add_method(self):
        """Testing the method for saving graph data"""
        species = [Specie("c1ccccc1"), Specie("O=O"), Specie("N")]
        t1 = time()
        _, ntrails = self.make_graph(1000, species, max_changing_bonds=3)
        t2 = time()
        print("total time:", t2 - t1)
        print("n trails:", ntrails)
        print("time per add:", (t2 - t1) / ntrails)

    @unittest.skip("")
    def test_new_method(self):
        g = RxnGraph()
        g.new()
    
    @unittest.skip("")
    def test_save_method(self):
        species = [Specie("c1ccccc1"), Specie("O=O"), Specie("N")]
        graph, ntrails = self.make_graph(100, species, max_changing_bonds=3)
        # adding some fictional source species
        graph.set_source_species(species, force=True)
        graph.save("test.rxn")
        os.remove("test.rxn")
    
    @unittest.skip("")
    def test_load_method(self):
        species = [Specie("c1ccccc1"), Specie("O=O"), Specie("N")]
        graph, ntrails = self.make_graph(30, species, max_changing_bonds=3)
        # adding some fictional source species
        graph.set_source_species(species, force=True)
        # saving
        graph.save("test.rxn")
        # loading
        loaded = RxnGraph.from_file("test.rxn")
        # comparing
        self.assertTrue(graph == loaded)
        # cleaning
        os.remove("test.rxn")

    def test_remove_method(self):
        from numpy.random import choice
        species = [Specie("c1ccccc1"), Specie("O=O"), Specie("N")]
        graph, ntrails = self.make_graph(30, species, max_changing_bonds=3)
        # adding some fictional source species
        graph.set_source_species(species, force=True)
        # choosing random specie
        while True:
            specie = choice(list(graph.species))
            if specie not in graph.source_species:
                break
        # removing specie
        print("Removing", specie.identifier)
        print("BEFORE")
        print("N Species", graph.get_n_species())
        print("N Reactions", graph.get_n_reactions())
        graph = graph.remove_specie(specie)
        print("AFTER")
        print("N Species", graph.get_n_species())
        print("N Reactions", graph.get_n_reactions())
        # making sure node doesnt exist anymore
        self.assertFalse(graph.specie_collection.has(specie))
        # making sure source species are still in graph
        self.assertTrue(all([graph.specie_collection.has(s) for s in graph.source_species]))

if __name__ == '__main__':
    unittest.main()
