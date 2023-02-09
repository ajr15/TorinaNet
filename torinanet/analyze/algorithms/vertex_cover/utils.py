import networkx as nx
from ....core.RxnGraph import RxnGraph
from ....core import Specie

def total_degree(rxn_graph: RxnGraph, G: nx.DiGraph, specie: Specie) -> int:
    """Greedy metric that calculates total degree of specie"""
    node = rxn_graph.specie_collection.get_key(specie)
    return len(list(nx.all_neighbors(G, node)))

def percolation_degree(rxn_graph: RxnGraph, G: nx.DiGraph, specie: Specie) -> int:
    """Greedy metric that calculates the number of removed reactions after removal of a specie"""
    before = len(rxn_graph.reactions)
    after = len(rxn_graph.remove_specie(specie).reactions)
    return before - after
