import networkx as nx
import unittest
import torinanet as tn
from tests.utils import specie_generator, reaction_generator

@unittest.skip("HI")
class TestCuckooSpecieCollection (unittest.TestCase):

    def make_collection(self, size):
        seed_species = [tn.core.Specie("CC"), tn.core.Specie("O=O"), tn.core.Specie("N")]
        print("Building collection with {} species".format(size))
        collection = tn.core.HashedCollection.CuckooSpecieCollection()
        trails = set()
        for specie in specie_generator(seed_species, 4):
            # print("Adding {}, n_species={}, n_trails={}".format(specie.identifier, len(collection), len(trails)))
            trails.add(specie.identifier)
            collection.add(specie)
            if len(collection) >= size:
                print("=== DONE ===")
                return collection, trails

    def test_true_isomorphism(self):
        """out of the molecules that were not inserted, do they really have an isomorph in the collection?"""
        # making a large collection
        collection, trails = self.make_collection(500)
        inserted = set([s.identifier for s in collection.objects()])
        print("Equivalent SMILES")
        # showing all smiles in trails that were not added because of equivalence
        print("{:<30} {}".format("SMILES", "Isomorphic"))
        for smiles in trails - inserted:
            sp = tn.core.Specie(smiles)
            sp = tn.core.BinaryAcMatrix.from_specie(sp).to_specie()
            isosp = collection.get(sp)
            # printing for human inspection
            print("{:<30} {}".format(sp.identifier, isosp.identifier))
            # running some autotests
            g1 = sp.ac_matrix.to_networkx_graph()
            g2 = isosp.ac_matrix.to_networkx_graph()
            # checking networkx isomorphism
            self.assertTrue(nx.is_isomorphic(g1, g2, node_match=lambda x, y: x["Z"] == y["Z"]))

    def test_no_duplicates(self):
        """inside the collection are all molecules unique? (no isomorphs)"""
        # making a large collection
        collection, trails = self.make_collection(500)
        # going over all species in collection & validating uniqueness
        species = list(collection.objects())
        for i, sp1 in enumerate(species):
            print(sp1.identifier)
            for sp2 in species[(i + 1):]:
                g1 = sp1.ac_matrix.to_networkx_graph()
                g2 = sp2.ac_matrix.to_networkx_graph()
                self.assertFalse(nx.is_isomorphic(g1, g2, node_match=lambda x, y: x["Z"] == y["Z"]), msg="specie {} is equivalent to {}".format(sp1.identifier, sp2.identifier))

class TestCuckooReactionCollection (unittest.TestCase):

    def make_collection(self, size):
        seed_species = [tn.core.Specie("CC"), tn.core.Specie("O=O"), tn.core.Specie("N")]
        print("Building collection with {} reactions".format(size))
        collection = tn.core.HashedCollection.IndepCuckooReactionCollection()
        trails = list()
        been_thru = set()
        for rxn in reaction_generator(seed_species, 4):
            # print("Adding {}, n_species={}, n_trails={}".format(specie.identifier, len(collection), len(trails)))
            s = self.rxn_string(rxn)
            if s not in been_thru:
                trails.append(rxn)
                been_thru.add(s)
            collection.add(rxn)
            if len(collection) >= size:
                print("=== DONE ===")
                return collection, trails

    @staticmethod
    def rxn_string(rxn):
        return "{} = {}".format(" + ".join([s.identifier for s in rxn.reactants]), " + ".join([s.identifier for s in rxn.products]))

    def test_true_isomorphism(self):
        """out of the reactions that were not inserted, do they really have an isomorph in the collection?"""
        # making a large collection
        collection, trails = self.make_collection(200)
        print("Equivalent SMILES")
        # showing all smiles in trails that were not added because of equivalence
        print("{:<30} {}".format("SMILES", "Isomorphic"))
        for rxn in trails:
            if not collection.has(rxn):
                s1 = self.rxn_string(rxn)
                s2 = self.rxn_string(collection.get(rxn))
                print("{:<60} {}".format(s1, s2))
                # checking networkx isomorphism
                self.assertTrue(rxn == collection.get(rxn), msg="reaction {} was assigned to differnt reaction {}".format(s1, s2))

    def test_no_duplicates(self):
        """inside the collection are all reactions unique? (no isomorphs)"""
        # making a large collection
        collection, trails = self.make_collection(200)
        # going over all species in collection & validating uniqueness
        species = list(collection.objects())
        for i, r1 in enumerate(species):
            for r2 in species[(i + 1):]:
                s1 = self.rxn_string(r1)
                s2 = self.rxn_string(r2)
                self.assertFalse(r1 == r2, msg="reactions {} and {} are equal".format(s1, s2))


if __name__ == "__main__":
    unittest.main()