import sys
sys.path.append("../..")
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

from torinanet.iterate.filters.ac_matrix_filters import MaxBondsPerAtom
from torinanet.iterate.filters.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds
from torinanet.iterate.stop_conditions import MaxIterNumber
from torinanet.core import Specie


def make_text_pos(pos, xshift, yshift):
    for k, v in pos.items():
        pos[k] = v - np.array([xshift, yshift])
    return pos


def visualize_rxn_graph(G):
    colormap = []
    sizemap = []
    for node in G:
        # print(node)
        if type(node) is str:
            colormap.append("blue")
            sizemap.append(100)
        else:
            colormap.append("green")
            sizemap.append(100)
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1)
    nx.draw_networkx_nodes(G, pos=pos, node_size=sizemap, node_color=colormap)
    nx.draw_networkx_edges(G, pos=pos, alpha=0.04)
    nx.draw_networkx_labels(G, pos=make_text_pos(pos, 0, 0.06), font_color='k')


def iterate_module_standard_test(rxn_graph_module, iterator_module):
    """Standard test for the iterate module. DO NOT CHANGE !"""
    print("Running standard test for the iterate module.")
    print("Calculating combinatoric network of H2O molecule dissociation.")
    print("Enumerating...")
    # making the test graph
    rxn_graph = rxn_graph_module()
    rxn_graph.add_specie(Specie("O"))
    stop_cond = MaxIterNumber(2)
    ac_filters = [MaxBondsPerAtom()]
    conversion_filters = [MaxChangingBonds(2), OnlySingleBonds()]
    iterator = iterator_module(rxn_graph)
    rxn_graph = iterator.enumerate_reactions(conversion_filters, ac_filters, stop_cond, verbose=1)
    # checking correctness of the result
    print("number of species check:", "PASS" if len(rxn_graph.species) == 8 else "FAIL")
    print("number of reactions check:", "PASS" if len(rxn_graph.reactions) == 9 else "FAIL")
    res_rxn_strings = [" + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.reactants]) + 
                    " -> " + 
                    " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.products])
                    for reaction in rxn_graph.reactions]
    correct_rxn_strings = [ "O -> [OH] + [H]", 
                            "O -> [O] + [H] + [H]",
                            "[OH] -> [O] + [H]",
                            "[OH] + [OH] -> OO",
                            "[OH] + [H] -> O",
                            "[OH] + [O] -> O[O]",
                            "[H] + [H] -> [H][H]",
                            "[H] + [O] -> [OH]",
                            "[O] + [O] -> O=O"]
    print("reactions test:", "PASS" if set(res_rxn_strings) == set(correct_rxn_strings) else "FAIL")
    visualize_rxn_graph(rxn_graph.to_networkx_graph())
    plt.show()

def charge_iterate_module_standard_test(rxn_graph_module, iterator_module):
    """Standard test for the iterate module. DO NOT CHANGE !"""
    from torinanet.iterate.iterators.ChargeIterator import ChargeIterator
    from torinanet.iterate.filters.charge_filters import MaxAbsCharge
    print("Running standard test for the iterate module.")
    print("Calculating combinatoric network of H2O molecule dissociation.")
    print("Enumerating...")
    # making the test graph
    rxn_graph = rxn_graph_module()
    s = Specie("O")
    s.charge = 0
    s = rxn_graph.add_specie(s)
    rxn_graph.set_source_species([s])
    stop_cond = MaxIterNumber(2)
    ac_filters = [MaxBondsPerAtom()]
    conversion_filters = [MaxChangingBonds(2), OnlySingleBonds()]
    iterator = iterator_module(rxn_graph)
    rxn_graph = iterator.enumerate_reactions(conversion_filters, ac_filters, stop_cond, verbose=1)
    # iterating charges
    print("Iterating charges")
    charge_iterator = ChargeIterator(rxn_graph, rxn_graph_module)
    charged_rxn_graph = charge_iterator.enumerate_charges(1, 0, [MaxAbsCharge(1)])
    # checking correctness of the result
    # print("number of species check:", "PASS" if len(rxn_graph.species) == 8 else "FAIL")
    # print("number of reactions check:", "PASS" if len(rxn_graph.reactions) == 9 else "FAIL")
    res_rxn_strings = [" + ".join(["[{}]{}".format(s.ac_matrix.to_specie().identifier.strip(), s.charge) for s in reaction.reactants]) + 
                    " -> " + 
                    " + ".join(["[{}]{}".format(s.ac_matrix.to_specie().identifier.strip(), s.charge) for s in reaction.products])
                    for reaction in charged_rxn_graph.reactions]
    print("\n".join(res_rxn_strings))
    
    # correct_rxn_strings = [ "O -> [OH] + [H]", 
    #                         "O -> [O] + [H] + [H]",
    #                         "[OH] -> [O] + [H]",
    #                         "[OH] + [OH] -> OO",
    #                         "[OH] + [H] -> O",
    #                         "[OH] + [O] -> O[O]",
    #                         "[H] + [H] -> [H][H]",
    #                         "[H] + [O] -> [OH]",
    #                         "[O] + [O] -> O=O"]
    # print("reactions test:", "PASS" if set(res_rxn_strings) == set(correct_rxn_strings) else "FAIL")
    # visualize_rxn_graph(rxn_graph.to_networkx_graph())
    # plt.show()

def main():
    import torinanet.iterate
    torinanet.iterate.kernel = "vanilla"
    from torinanet.iterate.iterators import Iterator
    from torinanet.core.RxnGraph import RxnGraph
    charge_iterate_module_standard_test(RxnGraph, Iterator)

if __name__ == '__main__':
    main()