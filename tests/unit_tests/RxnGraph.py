import sys; sys.path.append("../..")
import numpy as np
from matplotlib import pyplot as plt
from src.core.RxnGraph import RxnGraph
from src.core.Specie import Specie
from src.core.Reaction import Reaction
from src.core.AcMatrix.BinaryAcMatrix import BinaryAcMatrix

from src.Iterate.Iterator import Iterator
from src.Iterate.ac_matrix_filters import max_bonds_per_atom
from src.Iterate.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds
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
    max_itr = 2
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

if __name__ == '__main__':
    compare_species_test()