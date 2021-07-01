import sys; sys.path.append("../..")
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from src.Iterate.Iterator import Iterator
from src.Iterate.ac_matrix_filters import max_bonds_per_atom
from src.Iterate.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds
from src.core.Specie import Specie

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
            sizemap.append(700)
        else:
            colormap.append("green")
            sizemap.append(100)
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1)
    nx.draw_networkx_nodes(G, pos=pos, node_size=sizemap, node_color=colormap)
    nx.draw_networkx_edges(G, pos=pos, alpha=0.04)
    nx.draw_networkx_labels(G, pos=make_text_pos(pos, 0, 0.06), font_color='k')

	
def itr_test():
    # reactants: list, max_itr, ac_filters=[], conversion_filters=[])
    reactants = [Specie("C#C")]
    max_itr = 2
    ac_filters = [max_bonds_per_atom]
    conversion_filters = [MaxChangingBonds(4), OnlySingleBonds()]
    iterator = Iterator()

    # rxn_graph = iterator.gererate_rxn_network(reactants, max_itr, ac_filters, conversion_filters)
    rxn_graph = iterator.iterate_over_a_specie(reactants[0], ac_filters, conversion_filters)
    G = rxn_graph.to_netwokx_graph()
    visualize_rxn_graph(G)
    plt.show()


def rxn_add_test():
    import numpy as np
    from src.core.Reaction import Reaction
    from src.core.RxnGraph import RxnGraph
    from src.core.AcMatrix.BinaryAcMatrix import BinaryAcMatrix
    ac = np.array([[8, 1, 1], [1, 6, 1], [1, 1, 6]])
    ac = BinaryAcMatrix(ac)
    cmat1 = np.zeros((3, 3))
    cmat1[0, 1] = -1 
    cmat1[1, 0] = -1
    cmat2 = np.zeros((3, 3))
    cmat2[1, 2] = -1 
    cmat2[2, 1] = -1

    # set reaction graph
    rxn_graph = RxnGraph()
    # add reaction 1
    new_ac1 = ac.matrix + cmat1
    new_ac1 = BinaryAcMatrix(new_ac1)
    r1 = Reaction.from_ac_matrices(ac, new_ac1)
    print(r1.properties)
    rxn_graph.add_reaction(r1)
    print(len(rxn_graph.reactions))
    
    # add reaction 2
    new_ac2 = ac.matrix + cmat2
    new_ac2 = BinaryAcMatrix(new_ac2)
    r2 = Reaction.from_ac_matrices(ac, new_ac2)
    print(r1.properties)
    rxn_graph.add_reaction(r2)
    print(len(rxn_graph.reactions))



if __name__ == '__main__':
	itr_test()

