import sys; sys.path.append("../..")
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

from src.Iterate.Iterator import Iterator
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
    # specie1, specie2, ac_filters, conversion_filters
    rxn_graph = iterator.iterate_over_species(reactant1, reactant2, ac_filters, conversion_filters)
    # print(rxn_graph.species[1].ac_matrix.matrix)
    for r in rxn_graph.reactions:
        # print(r.properties)
        print(r.properties['r_num'], [s.identifier for s in r.reactants], r.properties['p_num'], [s.identifier for s in r.products])
        # print(r.products)
    # G = rxn_graph.to_netwokx_graph()
    # visualize_rxn_graph(G)
    # plt.show()
    return rxn_graph

def gererate_rxn_network_test(specie_smiles):
    # reactants: list, max_itr, ac_filters=[], conversion_filters=[])
    reactants = [Specie(specie_smiles)]
    max_itr = 2
    ac_filters = [max_bonds_per_atom]
    conversion_filters = [MaxChangingBonds(2), OnlySingleBonds()]
    iterator = Iterator()

    rxn_graph = iterator.gererate_rxn_network(reactants, max_itr, ac_filters, conversion_filters)
    # rxn_graph = iterator.iterate_over_a_specie(reactants[0], ac_filters, conversion_filters)
    for r in rxn_graph.reactions:
        # print(r.properties)
        print(r.properties['r_num'], [s.identifier for s in r.reactants], r.properties['p_num'], [s.identifier for s in r.products])
        # print(r.products)
    print("number of species", len(rxn_graph.species))
    print("number of reactions", len(rxn_graph.reactions))
    G = rxn_graph.to_netwokx_graph()
    visualize_rxn_graph(G)
    plt.show()

def main():
    # pre_g = iterate_over_a_specie_test("C#N")
    # for i, s in enumerate(pre_g.species):
    #     print("specie " + str(i))
    #     print(s.ac_matrix.matrix)
    # print("n reactions in pre:", len(pre_g.reactions))
    # print("n species in pre:", len(pre_g.species))
    gererate_rxn_network_test("[C+]=O")

if __name__ == '__main__':
    main()