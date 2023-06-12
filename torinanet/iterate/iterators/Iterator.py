import os
from abc import ABC
from time import time, sleep
from typing import List, Optional
from multiprocessing import Pool
from ...core.RxnGraph.RxnGraph import RxnGraph
from ..filters.conversion_matrix_filters import ConvFilter
from ..stop_conditions import StopCondition
from ...core.Reaction import Reaction
from ..filters.conversion_matrix_filters import _TwoSpecieMatrix
from ..kernels.commons import join_matrices
from ..kernels import utils


class Iterator (ABC):

    """Iterator object. Handles all elementary reacion enumeration operations.
    ARGS:
        - rxn_graph (RxnGraph): reaction graph object with specie information"""

    def __init__(self, rxn_graph: RxnGraph, n_workers: Optional[int]=None):
        self.rxn_graph = rxn_graph
        self.n_workers = n_workers if n_workers is None else os.cpu_count()

    @staticmethod
    def _assync_progress_listener(assync_results, verbose: int):
        """Listener on state of assync results, waits for completion & supports verbosity"""
        ti = time()
        prev_done = sum([res.ready() for res in assync_results]) 
        while True:
            sleep(1)
            if verbose > 0:
                done = sum([res.ready() for res in assync_results])
                if done > prev_done:
                    prev_done = done
                    print("{} done out of {} ({:.2f}%). total elapsed time {:.2f} seconds".format(done, len(assync_results), done / len(assync_results) * 100, time() - ti))
            if all([res.ready() for res in assync_results]):
                break
        if verbose > 0:
            print("all done in {} seconds".format(round(time() - ti)))

    # @TimeFunc
    @staticmethod
    def iterate_over_species(rxn_graph, specie1, specie2, ac_filters, conversion_filters):
        """iterate over all possible reactions for 2 species"""
        # create joint ac matrix
        ac1 = rxn_graph.ac_matrix_type.from_specie(specie1)
        ac2 = rxn_graph.ac_matrix_type.from_specie(specie2)
        joint_ac = ac1.__class__(join_matrices(ac1.matrix, ac2.matrix))
        max_l = max(len(ac1), len(ac2))
        # make new graph for results
        res_rxn_graph = rxn_graph.new(reactions=[], species=[])
        # adding original specie to the graph
        res_rxn_graph.add_specie(specie1)
        res_rxn_graph.add_specie(specie2)
        # iterate over all product AC matrices
        for ac in utils.enumerate_over_ac(joint_ac, conversion_filters + [_TwoSpecieMatrix(max_l)], ac_filters):
            # add to resulting reaction graph
            res_rxn_graph.add_reaction(Reaction.from_ac_matrices(joint_ac, ac)) 
        return res_rxn_graph


    # @TimeFunc
    @staticmethod
    def iterate_over_a_specie(rxn_graph, specie, ac_filters, conversion_filters):
        """iterate over all possible reactions for 2 species"""
        # create joint ac matrix
        origin_ac = rxn_graph.ac_matrix_type.from_specie(specie)
        res_rxn_graph = rxn_graph.new(reactions=[], species=[])
        # adding original specie to the graph
        res_rxn_graph.add_specie(specie)
        # apply all conversion matrices
        for ac in utils.enumerate_over_ac(origin_ac, conversion_filters, ac_filters):
            # add to resulting reaction graph
            res_rxn_graph.add_reaction(Reaction.from_ac_matrices(origin_ac, ac)) 
        return res_rxn_graph


    def enumerate_reactions(self, conversion_filters: List[ConvFilter], ac_filters: List[callable], stopping_condition: StopCondition, verbose=1) -> RxnGraph:
        """Method to enumerate over all elementary reactions available in reaction graph, until a stopping condition is met.
        ARGS:
            - conversion_filters (List[ConvFilter]): list of filters for conversion matrices
            - ac_filters (List[callable]): list of filters for AC matrices
        RETURNS:
            (RxnGraph) reaction graph with all enumerated reactions (based on self.rxn_graph)"""
        with Pool(self.n_workers) as pool:
            # init
            # specie sorting by size should make parallel running more efficient (no large species to enumerate at the end)
            old_species = sorted(list(self.rxn_graph.get_visited_species()), key=lambda s: len(s.ac_matrix), reverse=True)
            new_species = sorted(list(self.rxn_graph.get_unvisited_species()), key=lambda s: len(s.ac_matrix), reverse=True)
            # iterate reactions of species
            counter = 1
            while True:
                nseed = self.rxn_graph.new(reactions=[], species=[])
                # add all unimolecular reactions
                if verbose > 0:
                    print("=" * 30)
                    print(" " * 9 + "Iteration {}".format(counter))
                    print("=" * 30)
                    print("\n")
                    print("enumerating single-specie reactions...")
                ajr = []
                for s in new_species:
                    r = pool.apply_async(self.iterate_over_a_specie, (self.rxn_graph, s, ac_filters, conversion_filters))
                    ajr.append(r)
                # wait until completion of tasks + verbosity
                self._assync_progress_listener(ajr, verbose)
                for ajr_rxn_graph in ajr:
                    # do not forget to "get" the value from the AsyncResult
                    nseed.join(ajr_rxn_graph.get())
                if verbose > 0:
                    print("Current number of species:", nseed.get_n_species())
                    print("Current number of reactions:", nseed.get_n_reactions())
                    print("\n")
                    print("enumerating old-new specie reactions...")
                # add all bimolecular reactions
                # new species and old species
                ajr = []
                for s1 in new_species:
                    for s2 in old_species: 
                        r = pool.apply_async(self.iterate_over_species, (self.rxn_graph, s1, s2, ac_filters, conversion_filters))
                        ajr.append(r)
                self._assync_progress_listener(ajr, verbose)
                for ajr_rxn_graph in ajr:
                    # do not forget to "get" the value from the AsyncResult
                    nseed.join(ajr_rxn_graph.get())
                if verbose > 0:
                    print("Current number of species:", nseed.get_n_species())
                    print("Current number of reactions:", nseed.get_n_reactions())
                    print("\n")
                    print("enumerating new-new specie reactions...")
                # new species and new species
                ajr = []
                for i in range(len(new_species)):
                    for j in range(i, len(new_species)):
                        r = pool.apply_async(self.iterate_over_species, (self.rxn_graph, new_species[i], new_species[j], ac_filters, conversion_filters))
                        ajr.append(r)
                self._assync_progress_listener(ajr, verbose)
                for ajr_rxn_graph in ajr:
                    # do not forget to "get" the value from the AsyncResult
                    nseed.join(ajr_rxn_graph.get())
                if verbose > 0:
                    print("Current number of species:", nseed.get_n_species())
                    print("Current number of reactions:", nseed.get_n_reactions())
                    print("DONE ITERATING REACTIONS IN THIS ITERATION SUCCESSFULLY")
                # update rxn_graph
                self.rxn_graph.join(nseed)
                # make new species visited
                for specie in new_species:
                    self.rxn_graph.make_specie_visited(specie)
                # check stop condition
                stop = stopping_condition.check(self.rxn_graph, counter)
                if verbose > 0:
                    stopping_condition.check_msg(self.rxn_graph, counter)
                if stop:
                    return self.rxn_graph
                # updating species
                old_species = sorted(list(self.rxn_graph.get_visited_species()), key=lambda s: len(s.ac_matrix), reverse=True)
                new_species = sorted(list(self.rxn_graph.get_unvisited_species()), key=lambda s: len(s.ac_matrix), reverse=True)
                counter += 1