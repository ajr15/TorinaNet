import networkx as nx
from ....core.RxnGraph import BaseRxnGraph
from ....core import Specie

def total_degree(rxn_graph: BaseRxnGraph, G: nx.DiGraph, specie: Specie) -> int:
    """Greedy metric that calculates total degree of specie"""
    return len([s for s in nx.all_neighbors(G, rxn_graph.make_unique_id(specie))])

def percolation_degree(rxn_graph: BaseRxnGraph, G: nx.DiGraph, specie: Specie) -> int:
    """Greedy metric that calculates the number of removed reactions after removal of a specie"""
    before = len(rxn_graph.reactions)
    after = len(rxn_graph.remove_specie(specie).reactions)
    return before - after
