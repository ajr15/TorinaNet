import sys; sys.path.append("../..")
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from torinanet.core.RxnGraph.RxnGraph import RxnGraph
from torinanet.core.Specie import Specie
from torinanet.core.Reaction import Reaction
from torinanet.core.AcMatrix.BinaryAcMatrix import BinaryAcMatrix

from torinanet.iterate.iterators.VanillaIterator import Iterator
from torinanet.iterate.filters.ac_matrix_filters import max_bonds_per_atom
from torinanet.iterate.filters.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds
from Iterate import visualize_rxn_graph


# - to_netwokx_graph
# - compare_species
# - join
def gen_random_binary_ac_mats(number_of_ac_mats, ac_length):
    """Generates a set of reaction-like ac matrices. returns list of reactions from random ac matrices"""
    ac_mats = []
    counter = 0
    while len(ac_mats) < number_of_ac_mats:
        mat = np.zeros((ac_length, ac_length))
        for i in range(ac_length):
            mat[i][i] = np.random.choice([1, 1, 4]) # generating random C or H atom
            for j in range(i + 1, ac_length):
                # randomizing bonds (with ~30% to 70% ratio between 1 and 0)
                v = int(round(0.7 * np.random.rand()))
                mat[i][j] = v
                mat[j][i] = v
        mat = BinaryAcMatrix(mat)
        comps = mat.get_compoenents()
        if len(comps) == 2:
            rxn = Reaction.from_ac_matrices(*comps)
            ac_mats.append(rxn)
            # flips a coin for appending same matrix (deliberately causing equality)
            if np.random.rand() > 0.5:
                counter += 1
                ac_mats.append(rxn)
    print("Number of identical matrices:", counter)
    return ac_mats

def add_reaction_test():
    # some toy data - random binary ac matrices
    n_rxns = 8
    print("Number of reactions:", n_rxns)
    ac_mats = gen_random_binary_ac_mats(n_rxns, 4)
    rxn_graph = RxnGraph()
    for rxn in ac_mats:
        print("REACTANTS")
        print(rxn.reactants[0].ac_matrix.matrix)
        print(np.linalg.det(rxn.reactants[0].ac_matrix.matrix))
        print("PRODUCTS")
        print(rxn.products[0].ac_matrix.matrix)
        print(np.linalg.det(rxn.products[0].ac_matrix.matrix))
        print("PROPERTIES")
        print(rxn.properties)
        print("=" * 30)
        rxn_graph.add_reaction(rxn)
    print("Number of reactions in graph", len(rxn_graph.reactions))
    

def to_networkx_graph_test():
    """function to test the action of \'to_networkx_graph\' method in a reaction graph"""
    pass

def compare_species_test():
    """function to test the action of \'compare_species\' method in a reaction graph"""
    g1 = RxnGraph()
    s = Specie("C#N")
    g1.add_specie(Specie.from_ac_matrix(BinaryAcMatrix.from_specie(s)))
    g2 = iterate_over_a_smiles("C#N")
    print("number of species in g1", len(g1.species))
    print("number of species in g2", len(g2.species))
    print("comparing g1 to g2", len(g1.compare_species(g2)))
    print("comparing g2 to g1", len(g2.compare_species(g1)))


def iterate_over_a_smiles(specie_smiles):
    # reactants: list, max_itr, ac_filters=[], conversion_filters=[])
    reactants = [Specie(specie_smiles)]
    max_itr = 1
    ac_filters = [max_bonds_per_atom]
    conversion_filters = [MaxChangingBonds(4), OnlySingleBonds()]
    iterator = Iterator()

    # rxn_graph = iterator.gererate_rxn_network(reactants, max_itr, ac_filters, conversion_filters)
    rxn_graph = iterator.iterate_over_a_specie(reactants[0], ac_filters, conversion_filters)
    return rxn_graph
    
def iterate_over_smiles(specie_smiles1, specie_smiles2):
    # reactants: list, max_itr, ac_filters=[], conversion_filters=[])
    reactant1 = Specie(specie_smiles1)
    reactant2 = Specie(specie_smiles2)
    max_itr = 2
    ac_filters = [max_bonds_per_atom]
    conversion_filters = [MaxChangingBonds(4), OnlySingleBonds()]
    iterator = Iterator()
    # specie1, specie2, ac_filters, conversion_filters
    rxn_graph = iterator.iterate_over_species(reactant1, reactant2, ac_filters, conversion_filters)
    return rxn_graph

def join_test():
    """function to test the action of \'join\' method in a reaction graph"""
    # join two different graphs
    print("DIFFERENT GRAPHS !")
    g1 = iterate_over_a_smiles("C#N")
    g2 = iterate_over_a_smiles("C=N")
    print("n of reactions in g1:", len(g1.reactions))
    print("n of reactions in g2:", len(g2.reactions))
    print("n of species in g1:", len(g1.species))
    print("n of species in g2:", len(g2.species))
    visualize_rxn_graph(g1.to_netwokx_graph())
    g1.join(g2)
    print("n of reactions in joint graph:", len(g1.reactions))
    print("n of species in joint graph:", len(g1.species))
    visualize_rxn_graph(g1.to_netwokx_graph())


    # join identical graphs
    print("SAME GRAPHS !")
    g1 = iterate_over_a_smiles("C=C")
    g2 = iterate_over_a_smiles("C=C")
    print("n of reactions in g1:", len(g1.reactions))
    print("n of reactions in g2:", len(g2.reactions))
    print("n of species in g1:", len(g1.species))
    print("n of species in g2:", len(g2.species))
    visualize_rxn_graph(g1.to_netwokx_graph())
    g1.join(g2)
    print("n of reactions in joint graph:", len(g1.reactions))
    print("n of species in joint graph:", len(g1.species))
    visualize_rxn_graph(g1.to_netwokx_graph())
    plt.show()

def join_empty_test():
    g1 = RxnGraph()
    g1.add_specie(Specie("C#N"))
    g2 = iterate_over_a_smiles("C#N")
    print("n of reactions in g1:", len(g1.reactions))
    print("n of reactions in g2:", len(g2.reactions))
    print("n of species in g1:", len(g1.species))
    print("n of species in g2:", len(g2.species))
    visualize_rxn_graph(g1.to_netwokx_graph())
    visualize_rxn_graph(g2.to_netwokx_graph())
    g1.join(g2)
    print("n of reactions in joint graph:", len(g1.reactions))
    print("n of species in joint graph:", len(g1.species))
    visualize_rxn_graph(g1.to_netwokx_graph())
    plt.show()

def generate_rxn_network_test(specie_smiles):
    reactants = [Specie(specie_smiles)]
    max_itr = 2
    ac_filters = [max_bonds_per_atom]
    conversion_filters = [MaxChangingBonds(2), OnlySingleBonds()]
    iterator = Iterator()

    rxn_graph = iterator.gererate_rxn_network(reactants, max_itr, ac_filters, conversion_filters)
    return rxn_graph


def save_graph_test():
    rxn_graph = generate_rxn_network_test("C#N")
    print("no. of species (before) =", len(rxn_graph.species))
    smiles_set = set()
    for s in rxn_graph.species:
        smiles = s.ac_matrix.to_specie().identifier
        print("=" * 30)
        print(s.ac_matrix.matrix)
        print(smiles)
        if not smiles in smiles_set:
            smiles_set.add(smiles)
        else:
            print("DOUBLE !")
            
    rxn_graph.save_to_file("./rxn_graph.dat")

    print("no. of species (after) =", len(rxn_graph.species))

def make_text_pos(pos, xshift, yshift):
    for k, v in pos.items():
        pos[k] = v - np.array([xshift, yshift])
    return pos


def plot_rxn_graph(G: RxnGraph, keep_ids, reactant_ids):
    colormap = []
    sizemap = []
    g = G.to_netwokx_graph(use_internal_id=False)
    for node in G.to_netwokx_graph(use_internal_id=True):
        # blue nodes = reactant nodes
        # green nodes = specie nodes (not deleted)
        # black nodes = reaction nodes (not deleted)
        # red nodes (small) = deleted reaction nodes
        # red nodes (large) = deleted specie nodes
        if node in reactant_ids:
            colormap.append("blue")
            sizemap.append(100)
        elif node in keep_ids:
            if node in G._specie_ids:
                colormap.append("green")
                sizemap.append(100)
            else:
                colormap.append("black")
                sizemap.append(50)
        else:
            if node in G._specie_ids:
                colormap.append("red")
                sizemap.append(100)
            else:
                colormap.append("red")
                sizemap.append(30)
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(g, k=1)
    nx.draw_networkx_nodes(g, pos=pos, node_size=sizemap, node_color=colormap)
    nx.draw_networkx_edges(g, pos=pos, alpha=0.04)
    nx.draw_networkx_labels(g, pos=make_text_pos(pos, 0, 0.06), font_color='k')


def remove_test():
    specie = BinaryAcMatrix.from_specie(Specie("[OH]")).to_specie()
    rxn_graph = generate_rxn_network_test("O")
    ids = rxn_graph._dfs_remove_id_str(rxn_graph.to_netwokx_graph(use_internal_id=True), specie._get_id_str(), rxn_graph.reactant_species)
    ids = ids.union(specie._get_id_str())
    plot_rxn_graph(rxn_graph, ids, [s._get_id_str() for s in rxn_graph.reactant_species])
    print("no. of species (before) =", len(rxn_graph.species))
    print("no. of reactions (before) =", len(rxn_graph.reactions))
    res_rxn_strings = [" + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.reactants]) + 
            " -> " + 
            " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.products])
            for reaction in rxn_graph.reactions]
    print("\n".join(res_rxn_strings))

    rxn_graph = rxn_graph.remove_specie(specie)
    print("no. of species (after) =", len(rxn_graph.species))
    print("no. of reactions (after) =", len(rxn_graph.reactions))
    res_rxn_strings = [" + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.reactants]) + 
                " -> " + 
                " + ".join([s.ac_matrix.to_specie().identifier.strip() for s in reaction.products])
                for reaction in rxn_graph.reactions]
    print("\n".join(res_rxn_strings))

    plt.show()

if __name__ == '__main__':
    remove_test()