# a test to compare the speed of hash-based RxnGraph add_specie and add_reaction vs linear search methods
import sys; sys.path.append("../..")
import numpy as np
from matplotlib import pyplot as plt
from time import time
from src.core.RxnGraph.RxnGraph import RxnGraph as hashGraph
from src.core.RxnGraph.vanillaRxnGraph import RxnGraph as linearGraph
from src.core.Reaction import Reaction
from src.core.AcMatrix.BinaryAcMatrix import BinaryAcMatrix
from src.Iterate.Iterator import Iterator

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

def add_reaction_time(ac_mats, search_method):
    # some toy data - random binary ac matrices
    n_rxns = 8
    if search_method == 'linear':
        rxn_graph = linearGraph()
    elif search_method == 'hash':
        rxn_graph = hashGraph()
    t1 = time()
    for rxn in ac_mats:
        rxn_graph.add_reaction(rxn)
    t2 = time()
    return t2 - t1

if __name__ == '__main__':
    t_hashes = []
    t_lins = []
    ns = [100, 1e3, 1e4]
    for n in ns:
        ac_mats = gen_random_binary_ac_mats(n, 6)
        thash = round(add_reaction_time(ac_mats, 'hash'), 2)
        tlin = round(add_reaction_time(ac_mats, 'linear'), 2)
        t_hashes.append(thash)
        t_lins.append(tlin)
    plt.plot(ns, t_hashes, 'r-', label='hash')
    plt.plot(ns, t_lins, 'k-', label='linear')
    plt.xlabel("Number of reactions")
    plt.ylabel("Time for adding (seconds)")
    plt.legend()
    plt.show()
    