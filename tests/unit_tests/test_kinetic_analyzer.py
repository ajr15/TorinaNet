from config import TORINA_NET_DIR, TORINA_X_DIR
import sys
sys.path.append(TORINA_NET_DIR)
sys.path.append(TORINA_X_DIR)
import matplotlib.pyplot as plt
import numpy as np
import unittest
import torinanet as tn

def make_rxn_strings(rxn_graph: tn.core.RxnGraph):
    strs = set()
    for reaction in rxn_graph.reactions:
        s = " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.reactants]) + \
                " -> " + \
                " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.products]) + \
                " k = {}".format(reaction.properties["k"])
        strs.add(s)
    return "\n".join(list(strs))

class TestKineticAnalyzer (unittest.TestCase):

    def setUp(self) -> None:
        """Sets up a default graph for testing - decomposition of H2O molecule"""
        # setting up graph
        print("Setting up reaction graph...")
        print("Using network of H2O decomposition")
        self.rxn_graph = tn.core.RxnGraph()
        self.h2o = self.rxn_graph.add_specie(tn.core.Specie("O"))
        self.oh = self.rxn_graph.add_specie(tn.core.Specie("[OH]"))
        self.h = self.rxn_graph.add_specie(tn.core.Specie("[H]"))
        self.h2o2 = self.rxn_graph.add_specie(tn.core.Specie("OO"))
        self.reactions = [
            tn.core.Reaction([self.h2o], [self.oh, self.h], properties={"k": 1}),
            tn.core.Reaction([self.h, self.oh], [self.h2o], properties={"k": 0.1}),
            tn.core.Reaction([self.oh, self.oh], [self.h2o2], properties={"k": 0.5})
        ]
        for r in self.reactions:
            self.rxn_graph.add_reaction(r)
        print("reactions set:")
        print(make_rxn_strings(self.rxn_graph))
        print("DONE SETUP")

    def test_kinetics(self):
        """Testing the method for saving graph data"""
        print("Testing the KineticAnalyzer solver")
        nsteps = 100
        analyzer = tn.analyze.kinetics.KineticAnalyzer(self.rxn_graph)
        analyzer.solve_kinetics(nsteps, 0.1, np.array([1, 0, 0, 0]), name="lsoda", atol=1e-4, rtol=1e-6)
        h2o_c = []
        h2o2_c = []
        oh_c = []
        h_c = []
        for step in range(nsteps):
            h2o_c.append(analyzer.get_concentration(self.h2o, step))
            h2o2_c.append(analyzer.get_concentration(self.h2o2, step))
            oh_c.append(analyzer.get_concentration(self.oh, step))
            h_c.append(analyzer.get_concentration(self.h, step))
        plt.plot(range(nsteps), h2o_c, label="h2o")
        plt.plot(range(nsteps), h2o2_c, label="h2o2")
        plt.plot(range(nsteps), h_c, label="h")
        plt.plot(range(nsteps), oh_c, label="oh")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()