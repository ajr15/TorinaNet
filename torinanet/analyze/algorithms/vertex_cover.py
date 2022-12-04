import networkx as nx
import numpy as np
from typing import List, Optional
from ...core.RxnGraph.BaseRxnGraph import BaseRxnGraph
from ...core.Specie import Specie

def total_degree(rxn_graph: BaseRxnGraph, G: nx.DiGraph, specie: Specie) -> int:
    """Greedy metric that calculates total degree of specie"""
    return len([s for s in nx.all_neighbors(G, rxn_graph.make_unique_id(specie))])

def percolation_degree(rxn_graph: BaseRxnGraph, G: nx.DiGraph, specie: Specie) -> int:
    """Greedy metric that calculates the number of removed reactions after removal of a specie"""
    before = len(rxn_graph.reactions)
    after = len(rxn_graph.remove_specie(specie).reactions)
    return before - after

def greedy_mvc(rxn_graph: BaseRxnGraph, greedy_metric: str, only_products: bool=False,
               covered_reactions: Optional[List[str]]=None, covered_species: Optional[List[str]]=None, 
               max_samples: Optional[int]=None, verbose=1) -> List[Specie]:
    """Greedy algorithm for finding the MVC of a reaction graph.
    ARGS:
        - rxn_graph (TorinaNet.RxnGraph): reaction graph for analysis
        - greedy_metric (str): name of greedy mvc metric
        - only_products (bool): weather to use only the products of the reactions for the cover. default=False
        - covered_reactions (List[str]): list of reaction-IDs that are "covered". default=None
        - covered_species (List[str]): list of specie-IDs that are covered - won't be selected for cover. default=None
        - max_samples (int): max number of samples to draw when looking for MVC. default=None - no sample limit
        - verbose (int): set verbosity mode of function.
    RETURNS:
        (List[Specie]) list of MVC species"""
    # init
    _rxn_graph = rxn_graph.copy(use_charge=rxn_graph.use_charge)
    n_reactions = _rxn_graph.get_n_reactions()
    # defining set of covered reaction
    if covered_reactions:
        covered_reactions = set(covered_reactions)
    else:
        covered_reactions = set()
    # adding covered reactions from covered species
    if covered_species:
        for rxn in _rxn_graph.reactions:
            if all([_rxn_graph.make_unique_id(s) in covered_species for
                    s in rxn.reactants + rxn.products]):
                covered_reactions.add(rxn_graph.make_unique_id(rxn))
    else:
        covered_species = []
    # setting greedy metric
    metrics = {"percolation": percolation_degree, "degree": total_degree}
    if not greedy_metric.lower() in metrics:
        raise ValueError("Unknown greedy_metric {}, allowed options {}".format(greedy_metric.lower(), ", ".join(metrics.keys())))
    else:
        greedy_metric = metrics[greedy_metric]
    sample_counter = 0
    mvc = set()
    # main loop
    while True:
        # stopping condition = no more uncovered reactions => remained reactions = uncovered reactions
        sample_counter += 1
        remained_reactions = set([_rxn_graph.make_unique_id(r) for r in _rxn_graph.reactions])
        if remained_reactions.issubset(covered_reactions):
            if verbose > 0:
                print("Found MVC !")
            return [rxn_graph.get_specie_from_id(s) for s in list(mvc)]

        if verbose > 0:
            n_left_reactions = len(list(remained_reactions.difference(covered_reactions))) 
            print("covered {} out of {} ({:.2f}%)".format(n_reactions - n_left_reactions,
                                                          n_reactions, (1 - n_left_reactions / n_reactions) * 100))
        # randomly select reaction (all are uncovered)
        rxn = np.random.choice(_rxn_graph.reactions)
        if _rxn_graph.make_unique_id(rxn) in covered_reactions:
            continue
        # add to mvc product / reactant with greedy metric value
        G = _rxn_graph.to_networkx_graph(use_internal_id=True)
        max_x = 0
        specie = None
        ajr = rxn.products if only_products else rxn.reactants + rxn.products
        for s in ajr:
            if not rxn_graph.make_unique_id(s) in covered_species:
                x = greedy_metric(_rxn_graph, G, s)
                if x >= max_x and not s in _rxn_graph.source_species:
                    max_x = x
                    specie = s
        if not specie:
          continue
        mvc.add(rxn_graph.make_unique_id(specie))
        # clear the network from dependent reactions
        _rxn_graph = _rxn_graph.remove_specie(s)
        # checking for max samples
        if not max_samples is None and sample_counter >= max_samples:
            if verbose > 0:
                print("MVC not found after {} samples".format(max_samples))
            return []

def max_metric_mvc(rxn_graph: BaseRxnGraph, greedy_metric: str, covered_reactions: Optional[List[str]]=None, 
                    covered_species: Optional[List[str]]=None, verbose=1) -> List[Specie]:
    """Find MVC deterministically by taking the uncovered species with maximum metric values"""
    # init
    _rxn_graph = rxn_graph.copy(use_charge=rxn_graph.use_charge)
    n_reactions = _rxn_graph.get_n_reactions()
    # defining set of covered reaction
    if covered_reactions:
        covered_reactions = set(covered_reactions)
    else:
        covered_reactions = set()
    # adding covered reactions from covered species
    if covered_species:
        for rxn in _rxn_graph.reactions:
            if all([_rxn_graph.make_unique_id(s) in covered_species for
                    s in rxn.reactants + rxn.products]):
                covered_reactions.add(rxn_graph.make_unique_id(rxn))
    else:
        covered_species = []
    # setting greedy metric
    metrics = {"percolation": percolation_degree, "degree": total_degree}
    if not greedy_metric.lower() in metrics:
        raise ValueError("Unknown greedy_metric {}, allowed options {}".format(greedy_metric.lower(), ", ".join(metrics.keys())))
    else:
        greedy_metric = metrics[greedy_metric]
    sample_counter = 0
    mvc = set()
    # main loop
    while True:
        # stopping condition = no more uncovered reactions => remained reactions = uncovered reactions
        sample_counter += 1
        remained_reactions = set([_rxn_graph.make_unique_id(r) for r in _rxn_graph.reactions])
        if remained_reactions.issubset(covered_reactions):
            if verbose > 0:
                print("Found MVC !")
            return [rxn_graph.get_specie_from_id(s) for s in list(mvc)]

        if verbose > 0:
            n_left_reactions = len(list(remained_reactions.difference(covered_reactions))) 
            print("covered {} out of {} ({:.2f}%)".format(n_reactions - n_left_reactions,
                                                          n_reactions, (1 - n_left_reactions / n_reactions) * 100))
        # select uncovered reaction with maximum metric value
        G = _rxn_graph.to_networkx_graph(use_internal_id=True)
        max_metric = 0
        specie = None
        for s in _rxn_graph.species:
            sid = rxn_graph.make_unique_id(s)
            if not sid in covered_species:
                x = greedy_metric(_rxn_graph, G, s)
                if x >= max_metric and not s in _rxn_graph.source_species:
                    max_metric = x
                    specie = s
        # if no specie is found, we have an mvc!
        if specie is None:
            if verbose > 0:
                print("Found MVC !")
            return [rxn_graph.get_specie_from_id(s) for s in list(mvc)]
        # else, add specie to MVC
        else:
            mvc.add(rxn_graph.make_unique_id(specie))
            # clear the network from dependent reactions
            _rxn_graph = _rxn_graph.remove_specie(specie)
