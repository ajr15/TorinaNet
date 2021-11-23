import sys; sys.path.append("../..")
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

from src.utils.TimeFunc import show_time_data
from src.Iterate.daskIterator import Iterator
from src.Iterate.ac_matrix_filters import max_bonds_per_atom
from src.Iterate.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds
from src.core.Specie import Specie
from src.core.AcMatrix.BinaryAcMatrix import BinaryAcMatrix

# - iterate_over_a_specie
# - iterate_over_species
# - generate_conversion_matrices
# - gererate_rxn_network

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

	
def iterate_over_a_specie_test(specie_smiles):
    # reactants: list, max_itr, ac_filters=[], conversion_filters=[])
    reactants = [Specie(specie_smiles)]
    max_itr = 2
    ac_filters = [max_bonds_per_atom]
    conversion_filters = [MaxChangingBonds(4), OnlySingleBonds()]
    iterator = Iterator()

    # rxn_graph = iterator.gererate_rxn_network(reactants, max_itr, ac_filters, conversion_filters)
    rxn_graph = iterator.iterate_over_a_specie(reactants[0], ac_filters, conversion_filters)
    for r in rxn_graph.reactions:
        # print(r.properties)
        print(r.properties['r_num'], r.reactants[0].identifier, r.properties['p_num'], [s.identifier for s in r.products])
        # print(r.products)

    G = rxn_graph.to_netwokx_graph()
    visualize_rxn_graph(G)
    plt.show()
    return rxn_graph

def generate_conversion_matrices_test(specie_smiles):
    ac = BinaryAcMatrix.from_specie(Specie(specie_smiles))
    conversion_filters = [MaxChangingBonds(4), OnlySingleBonds()]
    iterator = Iterator()

    cmats = iterator.generate_conversion_matrices(ac, conversion_filters)
    for i in range(len(cmats)):
        for j in range(i + 1, len(cmats)):
            if np.array_equal(cmats[i], cmats[j]):
                print(cmats[i])
                print(cmats[j])
                raise Exception("There are identical conversion matrices !!")

def iterate_over_species_test(specie_smiles1, specie_smiles2):
    # reactants: list, max_itr, ac_filters=[], conversion_filters=[])
    reactant1 = Specie(specie_smiles1)
    reactant2 = Specie(specie_smiles2)
    max_itr = 2
    ac_filters = [max_bonds_per_atom]
    conversion_filters = [MaxChangingBonds(4), OnlySingleBonds()]
    iterator = Iterator()
    rxn_graph = iterator.iterate_over_species(reactant1, reactant2, ac_filters, conversion_filters)
    return rxn_graph

def iterate_module_standard_test():
    """Standard test for the iterate module. DO NOT CHANGE !"""
    print("Running standard test for the Iterate module.")
    print("Calculating combinatoric network of H2O molecule dissociation.")
    print("Enumerating...")
    # making the test graph
    reactants = [Specie("O")]
    max_itr = 2
    ac_filters = [max_bonds_per_atom]
    conversion_filters = [MaxChangingBonds(2), OnlySingleBonds()]
    iterator = Iterator()
    rxn_graph = iterator.gererate_rxn_network(reactants, max_itr, ac_filters, conversion_filters, verbose=1)
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
    visualize_rxn_graph(rxn_graph.to_netwokx_graph())
    plt.show()

def main():
    iterate_module_standard_test()

if __name__ == '__main__':
    main()