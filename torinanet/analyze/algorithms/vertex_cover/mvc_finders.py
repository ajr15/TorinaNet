import numpy as np
from abc import ABC, abstractclassmethod
from typing import List, Optional
from ....core.RxnGraph.RxnGraph import RxnGraph
from ....core.Specie import Specie
from ....core.Reaction import Reaction
from .utils import total_degree, percolation_degree


class MvcFinder (ABC):
    
    @abstractclassmethod
    def step(self, rxn_graph: RxnGraph, uncovered_reactions: List[Reaction], mvc: List[Specie]) -> Specie:
        """One step of MVC finder, returns the next specie to add to the MVC"""
        pass

    def is_covered(self, rxn: Reaction, mvc: Optional[List[Specie]]=None) -> bool:
        """Method to check if a reaction in graph is covered"""
        # default criteria - if all reactants are covered and at least one product is covered
        if mvc is None:
            _mvc = []
        else:
            _mvc = mvc
        return any([s.properties["visited"] or s in _mvc for s in rxn.products]) and all([s.properties["visited"] or s in _mvc for s in rxn.reactants])
            
    def stop_condition(self, rxn_graph: RxnGraph, uncovered_reactions: List[Reaction], mvc: List[Specie]) -> bool:
        """Custom stop condition to exit the algorithm. Applied after the check that all reactions are covered. If the condition is met, the returned MVC is None"""
        return False

    def cleanup(self):
        """Method to call for cleanup after MVC search for reusablility of the object"""
        pass

    def find_mvc(self, rxn_graph: RxnGraph, verbose: int=1) -> List[str]:
        """Method to find a minimal vertex cover for a reaction network"""
        mvc = []
        uncovered_reactions = [rxn for rxn in rxn_graph.reactions if not self.is_covered(rxn)]
        if verbose:
            n_reactions = len(list(rxn_graph.reactions))
            n_uncovered = len(uncovered_reactions)
            print("Finding MVC for network, starting with {} covered out of {} ({:.2f}%)".format(n_reactions - n_uncovered, n_reactions, (n_reactions - n_uncovered) / n_reactions * 100))
        while True:
            # add new specie to MVC
            mvc.append(self.step(rxn_graph, uncovered_reactions, mvc))
            # updates the uncovered reactions list
            uncovered_reactions = [rxn for rxn in uncovered_reactions if not self.is_covered(rxn, mvc)]
            # adding verbosity
            if verbose:
                n_reactions = len(rxn_graph.reactions)
                n_uncovered = len(uncovered_reactions)
                print("Finding MVC for network, starting with {} covered out of {} ({:.2f}%)".format(n_reactions - n_uncovered, n_reactions, (n_reactions - n_uncovered) / n_reactions * 100))
            # checks if there are still uncovered reactions
            if len(uncovered_reactions) == 0:
                self.cleanup()
                return mvc
            # checks custom stop condition
            elif self.stop_condition(rxn_graph, uncovered_reactions, mvc):
                if verbose:
                    print("Custom stop condition appolied, exiting without mvc")
                self.cleanup()
                return None


class StochasticMvcFinder (MvcFinder):

    metric_dict = {
        "degree": total_degree,
        "percolation": percolation_degree
    }
    
    def __init__(self, greedy_metric: str, max_samples: int, only_products: bool):
        self.metric = self.metric_dict[greedy_metric]
        self.max_samples = max_samples
        self.only_products = only_products
        self.sample_counter = 0

    def cleanup(self):
        self.sample_counter = 0

    def stop_condition(self, rxn_graph: RxnGraph, uncovered_reactions: List[Reaction], mvc: List[Specie]) -> bool:
        if self.sample_counter >= self.max_samples:
            return True
        else:
            return False

    def step(self, rxn_graph: RxnGraph, uncovered_reactions: List[Reaction], mvc: List[Specie]) -> Specie:
        while True:
            self.sample_counter += 1
            rxn = np.random.choice(uncovered_reactions)
            ajr = rxn.products if self.only_products else rxn.reactants + rxn.products
            max_x = 0
            specie = None
            for s in ajr:
                if not s in mvc:
                    G = rxn_graph.to_networkx_graph(use_internal_id=True)
                    x = self.metric(rxn_graph, G, s)
                    if x >= max_x and not s in rxn_graph.source_species:
                        max_x = x
                        specie = s
            if not specie is None:
                return specie
        



class GreedyMvcFinder (MvcFinder):

    metric_dict = {
        "degree": total_degree,
        "percolation": percolation_degree
    }
    
    def __init__(self, greedy_metric: str):
        self.metric = self.metric_dict[greedy_metric]


    def step(self, rxn_graph: RxnGraph, uncovered_reactions: List[Reaction], mvc: List[Specie]) -> Specie:
        max_metric = 0
        specie = None
        G = rxn_graph.to_networkx_graph(use_internal_id=True)
        for rxn in uncovered_reactions:
            for s in rxn.products:
                if not s in mvc:
                    x = self.metric(rxn_graph, G, s)
                    if x >= max_metric and not s in rxn_graph.source_species:
                        max_metric = x
                        specie = s
        return specie