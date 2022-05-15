import networkx as nx
import numpy as np
from typing import List, Optional
from ...core.RxnGraph.BaseRxnGraph import BaseRxnGraph
from ...core.Specie import Specie

def total_degree(rxn_graph: BaseRxnGraph, G: nx.DiGraph, specie: Specie) -> int:
    """Greedy metric that calculates total degree of specie"""
    return len([s for s in nx.all_neighbors(G, specie.identifier)])

def percolation_degree(rxn_graph: BaseRxnGraph, G: nx.DiGraph, specie: Specie) -> int:
    """Greedy metric that calculates the number of removed reactions after removal of a specie"""
    before = len(rxn_graph.reactions)
    after = len(rxn_graph.remove_specie(specie).reactions)
    return before - after

def greedy_mvc(rxn_graph: BaseRxnGraph, greedy_metric: str, covered_reactions: Optional[List[str]]=None, verbose=1) -> List[Specie]:
    """Greedy algorithm for finding the MVC of a reaction graph.
    ARGS:
        - rxn_graph (TorinaNet.RxnGraph): reaction graph for analysis
        - greedy_metric (str): name of greedy mvc metric
        - verbose (int): set verbosity mode of function.
    RETURNS:
        (List[Specie]) list of MVC species"""
    # init
    _rxn_graph = rxn_graph.copy()
    n_reactions = _rxn_graph.get_n_reactions()
    if covered_reactions:
        covered_reactions = set(covered_reactions)
    else:
        covered_reactions = set()
    # setting greedy metric
    metrics = {"percolation": percolation_degree, "degree": total_degree}
    if not greedy_metric.lower() in metrics:
        raise ValueError("Unknown greedy_metric {}, allowed options {}".format(greedy_metric.lower(), ", ".join(metrics.keys())))
    else:
        greedy_metric = metrics[greedy_metric]
    mvc = set()
    # main loop
    while True:
        # randomly select reaction (all are uncovered)
        rxn = np.random.choice(_rxn_graph.reactions)
        if _rxn_graph.make_unique_id(rxn) in covered_reactions:
            continue
        # add to mvc product / reactant with greedy metric value
        G = _rxn_graph.to_networkx_graph(use_internal_id=True)
        max_x = 0
        specie = ""
        for s in rxn.reactants + rxn.products:
            x = greedy_metric(_rxn_graph, G, s)
            if x > max_x and not s in _rxn_graph.reactant_species:
                max_x = x
                specie = s
        mvc.add(specie._get_id_str())
        if verbose > 0:
            print("covered {} out of {} ({:.2f}%)".format(n_reactions - len(_rxn_graph.reactions), n_reactions, (n_reactions - len(_rxn_graph.reactions)) / n_reactions))
        # clear the network from dependent reactions
        _rxn_graph = _rxn_graph.remove_specie(s)
        # stopping condition = no more uncovered reactions => remained reactions = uncovered reactions
        remained_reactions = set([_rxn_graph.make_unique_id(r) for r in _rxn_graph.reactions])
        if remained_reactions == covered_reactions:
            if verbose > 0:
                print("Found MVC !")
            return [rxn_graph.species[rxn_graph._specie_ids[s]] for s in list(mvc)]