import unittest
import torinanet as tn

class TestEnergyReduction (unittest.TestCase):

    def make_graph(self):
        """Make simple graph for testing:
        O -> [OH] + [H],    E=0.1Ha
        [OH] -> [O] + [H]   E=0.3Ha
        [O] + [H] -> [OH]   E=-0.3Ha
        [OH] + [OH] -> OO   E=-0.15Ha"""
        # making species
        self.h2o = tn.core.Specie("O", properties={"energy": -10})
        self.oh = tn.core.Specie("[OH]", properties={"energy": -6})
        self.h = tn.core.Specie("[H]", properties={"energy": -3.9})
        self.o = tn.core.Specie("[O]", properties={"energy": -1.8})
        self.h2o2 = tn.core.Specie("OO", properties={"energy": -12.15})
        # making reactions
        self.r1 = tn.core.Reaction([self.h2o], [self.h, self.oh])
        self.r2 = tn.core.Reaction([self.oh], [self.h, self.o])
        self.r3 = tn.core.Reaction([self.oh, self.oh], [self.h2o2])
        self.r4 = tn.core.Reaction([self.h, self.o], [self.oh])
        # building graph
        rxn_graph = tn.core.RxnGraph()
        rxn_graph.add_reaction(self.r1)
        rxn_graph.add_reaction(self.r2)
        rxn_graph.add_reaction(self.r3)
        rxn_graph.add_reaction(self.r4)
        rxn_graph.set_source_species([self.h2o])
        print("ORIGINAL REACTIONS")
        for r in rxn_graph.reactions:
            print(r.pretty_string(), "E =", tn.analyze.network_reduction.EnergyReduction.SimpleEnergyReduction.calc_reaction_energy(r))
        return rxn_graph

    def test_simple_reduction(self):
        reducer = tn.analyze.network_reduction.EnergyReduction.SimpleEnergyReduction(0.2, True, 0.35)
        original = self.make_graph()
        g = reducer.apply(original)
        # printing reactions
        print("AFTER REDUCTION")
        for r in g.reactions:
            print(r.pretty_string(), "E =", tn.analyze.network_reduction.EnergyReduction.SimpleEnergyReduction.calc_reaction_energy(r))
        # testing proper removal
        self.assertTrue(g.has_specie(self.h2o))
        self.assertTrue(g.has_specie(self.oh))
        self.assertTrue(g.has_specie(self.h))
        self.assertTrue(g.has_specie(self.h2o2))


if __name__ == "__main__":
    unittest.main()