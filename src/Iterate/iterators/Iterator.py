from abc import ABC, abstractclassmethod
from typing import List
import dask as da
from ...core.RxnGraph.BaseRxnGraph import BaseRxnGraph
from ..filters.conversion_matrix_filters import ConvFilter
from ..stop_conditions import StopCondition
from ...core.Reaction import Reaction
from ..filters.conversion_matrix_filters import _TwoSpecieMatrix
from ..kernels.commons import ProgressCallback, join_matrices
from ..kernels import utils


class Iterator (ABC):

    """Iterator object. Handles all elementary reacion enumeration operations.
    ARGS:
        - rxn_graph (RxnGraph): reaction graph object with specie information"""

    def __init__(self, rxn_graph: BaseRxnGraph):
        self.rxn_graph = rxn_graph


    @da.delayed(pure=False)
    def iterate_over_species(self, specie1, specie2, ac_filters, conversion_filters):
        """iterate over all possible reactions for 2 species"""
        # create joint ac matrix
        ac1 = self.rxn_graph.ac_matrix_type.from_specie(specie1)
        ac2 = self.rxn_graph.ac_matrix_type.from_specie(specie2)
        joint_ac = ac1.__class__(join_matrices(ac1.matrix, ac2.matrix))
        max_l = max(len(ac1), len(ac2))
        # make new graph for results
        res_rxn_graph = self.rxn_graph.__class__(**self.rxn_graph.to_dict())
        # adding original specie to the graph
        res_rxn_graph.add_specie(specie1)
        res_rxn_graph.add_specie(specie2)
        # iterate over all product AC matrices
        for ac in utils.enumerate_over_ac(joint_ac, conversion_filters + [_TwoSpecieMatrix(max_l)], ac_filters):
            # add to resulting reaction graph
            res_rxn_graph.add_reaction(Reaction.from_ac_matrices(joint_ac, ac)) 
        return res_rxn_graph


    @da.delayed(pure=False)
    def iterate_over_a_specie(self, specie, ac_filters, conversion_filters):
        """iterate over all possible reactions for 2 species"""
        # create joint ac matrix
        origin_ac = self.rxn_graph.ac_matrix_type.from_specie(specie)
        res_rxn_graph = self.rxn_graph.__class__(**self.rxn_graph.to_dict())
        # adding original specie to the graph
        res_rxn_graph.add_specie(specie)
        # apply all conversion matrices
        for ac in utils.enumerate_over_ac(origin_ac, conversion_filters, ac_filters):
            # add to resulting reaction graph
            res_rxn_graph.add_reaction(Reaction.from_ac_matrices(origin_ac, ac)) 
        return res_rxn_graph


    def enumerate_reactions(self, conversion_filters: List[ConvFilter], ac_filters: List[callable], stopping_condition: StopCondition, verbose=1) -> BaseRxnGraph:
        """Method to enumerate over all elementary reactions available in reaction graph, until a stopping condition is met.
        ARGS:
            - conversion_filters (List[ConvFilter]): list of filters for conversion matrices
            - ac_filters (List[callable]): list of filters for AC matrices
        RETURNS:
            (RxnGraph) reaction graph with all enumerated reactions (based on self.rxn_graph)"""
        # init
        old_species = self.rxn_graph.get_visited_species()
        new_species = self.rxn_graph.get_unvisited_species()
        # iterate reactions of species
        counter = 1
        while True:
            nseed = self.rxn_graph.__class__(**self.rxn_graph.to_dict())
            # add all unimolecular reactions
            if verbose > 0:
                print("=" * 30)
                print(" " * 9 + "Iteration {}".format(counter))
                print("=" * 30)
                print("\n")
                print("Iterating over single specie reactions...")
            ajr = []
            for s in new_species:
                ajr.append(self.iterate_over_a_specie(s, ac_filters, conversion_filters))
            if verbose > 0:
                with ProgressCallback():
                    ajr = da.compute(ajr, scheduler="processes")[0]
            else:
                ajr = da.compute(ajr, scheduler="processes")[0]
            for ajr_rxn_graph in ajr:
                nseed.join(ajr_rxn_graph)
            if verbose > 0:
                print("Current number of species:", nseed.get_n_species())
                print("Current number of reactions:", nseed.get_n_reactions())
                print("\n")
                print("Iterating over old-new specie reactions...")
            # add all bimolecular reactions
            # new species and old species
            ajr = []
            for s1 in new_species:
                for s2 in old_species: 
                    ajr.append(self.iterate_over_species(s1, s2, ac_filters, conversion_filters))
            if verbose > 0:
                with ProgressCallback():
                    ajr = da.compute(ajr, scheduler="processes")[0]
            else:
                ajr = da.compute(ajr, scheduler="processes")[0]
            for ajr_rxn_graph in ajr:
                nseed.join(ajr_rxn_graph)
            if verbose > 0:
                print("Current number of species:", nseed.get_n_species())
                print("Current number of reactions:", nseed.get_n_reactions())
                print("\n")
                print("iterating new-new specie reactions")
            # new species and new species
            ajr = []
            for i in range(len(new_species)):
                for j in range(i, len(new_species)): 
                    ajr.append(self.iterate_over_species(new_species[i], new_species[j], ac_filters, conversion_filters))
            if verbose > 0:
                with ProgressCallback():
                    ajr = da.compute(ajr, scheduler="processes")[0]
            else:
                ajr = da.compute(ajr, scheduler="processes")[0]
            for ajr_rxn_graph in ajr:
                nseed.join(ajr_rxn_graph)
            if verbose > 0:
                print("Current number of species:", nseed.get_n_species())
                print("Current number of reactions:", nseed.get_n_reactions())
                print("DONE ITERATING REACTIONS IN THIS ITERATION SUCCESSFULLY")
            # update rxn_graph
            self.rxn_graph.join(nseed)
            # make new species visited
            for specie in new_species:
                self.rxn_graph.make_specie_visited(specie)
            # updating species
            old_species = self.rxn_graph.get_visited_species()
            new_species = self.rxn_graph.get_unvisited_species()
            # check stop condition
            stop = stopping_condition.check(self.rxn_graph, counter)
            if verbose > 0:
                stopping_condition.check_msg(self.rxn_graph, counter)
            if stop:
                return self.rxn_graph
            counter += 1