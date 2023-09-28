from typing import Dict, Optional
import numpy as np
import networkx as nx
import pandas as pd
from itertools import chain
from ...core.Specie import Specie
from ...core.Reaction import Reaction
from ...core.RxnGraph import RxnGraph
from ..kinetics.KineticAnalyzer import KineticAnalyzer
from ..kinetics.utils import assign_maximal_rate_constants
from ..algorithms.ShortestPathAnalyzer import ShortestPathAnalyzer

class SimpleKineticsReduction: 

    """Class that implements simple reaction energy difference reduction"""

    def __init__(self, reaction_rate_th: float, 
                        rate_constant_property: str, 
                        simulation_time: float, 
                        timestep: float, 
                        estimate_max_constants: bool,
                        temperature: float=298,
                        reactant_concs: Optional[Dict[Specie, float]]=None, 
                        energy_conversion_factor: float=4.359744e-18, 
                        specie_energy_property_name: str="energy", 
                        **solver_kwargs):
        self.reaction_rate_th = reaction_rate_th
        self.rate_constant_property = rate_constant_property
        self.simulation_time = simulation_time
        self.timestep = timestep
        self.reactant_concs = reactant_concs
        self.solver_kwargs = solver_kwargs
        self.estimate_max_constants = estimate_max_constants
        self.temperature = temperature
        self.energy_conversion_factor = energy_conversion_factor
        self.specie_energy_property_name = specie_energy_property_name
        self.rate_constant_property_name = rate_constant_property
        
    
    def apply(self, rxn_graph: RxnGraph):
        # if required, estimate maximal rate constants
        if self.estimate_max_constants:
            rxn_graph = assign_maximal_rate_constants(rxn_graph, self.temperature, self.energy_conversion_factor, self.specie_energy_property_name, self.rate_constant_property)
        if not self.reactant_concs:
            self.reactant_concs = {rxn_graph.make_unique_id(s): 1 for s in rxn_graph.source_species}
        # initializing kinetic solver
        analyzer = KineticAnalyzer(rxn_graph, self.rate_constant_property)
        # building initial concentrations
        initial_concs = np.zeros(len(rxn_graph.species))
        for s, v in self.reactant_concs.items():
            initial_concs[analyzer.get_specie_index(s)] = v
        # solving kinetics
        max_rates = analyzer.find_max_reaction_rates(self.simulation_time, self.timestep, initial_concs, **self.solver_kwargs)
        # analyzing reaction rates
        for rxn in rxn_graph.reactions:
            # ensure reaction is in graph after some reductions
            if rxn_graph.has_reaction(rxn):
                rid = rxn_graph.make_unique_id(rxn)
                if max_rates[rid] < self.reaction_rate_th:
                    rxn_graph = rxn_graph.remove_reaction(rxn)
        return rxn_graph

class MolRankReduction:

    def __init__(self, rank_th: float, 
                        rate_constant_property: str, 
                        estimate_max_constants: bool,
                        target: str="species",
                        temperature: float=298,
                        activation_energy_scaling_factor: float=30,
                        energy_conversion_factor: float=4.359744e-18, 
                        specie_energy_property_name: str="energy"):
        self.rank_th = rank_th
        self.rate_constant_property = rate_constant_property
        self.estimate_max_constants = estimate_max_constants
        if target.lower() in ["species", "reactions"]:
            self.target = target.lower()
        else:
            raise ValueError("Unknown target {}. allowed targets are 'reacitons' or 'species'".format(target))
        self.temperature = temperature
        self.energy_conversion_factor = energy_conversion_factor / activation_energy_scaling_factor
        self.specie_energy_property_name = specie_energy_property_name
        self.rate_constant_property_name = rate_constant_property


    @staticmethod
    def _calc_total_out_rate(g: nx.DiGraph, sp: str):
        """Helper method to calculate the total out flux out of a specie node in a network"""
        return sum([g.nodes[rxn]["rate"] if "rate" in g.nodes[rxn] else 0 for rxn in g.successors(sp)])

    @staticmethod
    def _calc_rate_fraction(g: nx.DiGraph, rxn: str, sp: str):
        """Helper method to calculate the flux fraction of a specific reaction node from a specie node"""
        if "total_rate" not in g.nodes[sp]:
            return 0
        elif g.nodes[sp]["total_rate"] == 0:
            return 0
        else:
            return g.nodes[rxn]["rate"] / g.nodes[sp]["total_rate"]

    def rank_species(self, rxn_graph: RxnGraph, return_network: bool=False):
        """Method to calculate the PageRank metric of a specie"""
        # initializing - converting to networkx graph 
        g = rxn_graph.to_networkx_graph(use_internal_id=True)
        # initializing the MolRank dictionary with values for each specie in graph
        ajr = pd.DataFrame({"p": np.zeros(rxn_graph.get_n_species()), "visited": np.zeros(rxn_graph.get_n_species()), "specie": rxn_graph.species}, index=list(rxn_graph.specie_collection.keys()))
        # setting seed probabilities of 1 for all source species
        seed = [rxn_graph.specie_collection.get_key(sp) for sp  in rxn_graph.source_species]
        for key in seed:
            ajr.loc[key, "p"] = 1
            ajr.loc[key, "visited"] = True
        # ==== STARTING PAGERANK ====
        # we now make a loop over the graph, until all species are visited we do not stop
        while len(seed) > 0:
            # collecting species in "first layer" after seed
            nseed = set()
            # going all over successor reaction nodes from seed, if all reactants are covered, its products are in "first layer"
            for rxn in chain(*[g.successors(sp) for sp in seed]):
                if all([ajr.loc[r, "visited"] for r in g.predecessors(rxn)]):
                    # also estimating the rate of the reaction node
                    g.nodes[rxn]["rate"] = np.prod([ajr.loc[s,"p"] for s in g.predecessors(rxn)]) * g.nodes[rxn]["obj"].properties["k"]
                    for sp in g.successors(rxn):
                        if not ajr.loc[sp, "visited"]:
                            nseed.add(sp)
            # assigning total rates for seed species - needs to be after reaction rate estimation
            for sp in seed:
                g.nodes[sp]["total_rate"] = self._calc_total_out_rate(g, sp)
            # calculating nseed specie ranks
            for sp in list(nseed):
                val = 0
                for rxn in g.predecessors(sp):
                    # validating the reaction has calculated "rate" - this ensures that we calculate only reactions from the same layer
                    if "rate" in g.nodes[rxn]:
                        # taking the product of probability transitions
                        val += np.prod([self._calc_rate_fraction(g, rxn, sp) for sp in g.predecessors(rxn)])
                # assigning the rank to the specie + making it visited
                ajr.loc[sp, "visited"] = True
                ajr.loc[sp, "p"] = val
            # now putting nseed as seed
            seed = list(nseed)
        if return_network:
            # adding "p" information to the graph 
            for sp in ajr.index:
                g.nodes[sp]["p"] = ajr.loc[sp, "p"]
            return g
        else:
            return ajr
        
    def rank_reactions(self, rxn_graph: RxnGraph):
        # get the maximal distance from source as number of iterations for MolRank
        analyzer = ShortestPathAnalyzer(rxn_graph, prop_func=lambda rxn: 1)
        n_iterations = max(analyzer.shortest_path_table["dist"].values)
        # initializing - converting to networkx graph 
        g = rxn_graph.to_networkx_graph(use_internal_id=True)
        # initializing rank dataframe
        ajr = pd.DataFrame({"p": np.zeros(rxn_graph.get_n_reactions()), "rxn": rxn_graph.reactions}, index=list(rxn_graph.reaction_collection.keys()))
        # setting the rank of all species to 0 and source species to 1
        for sp in rxn_graph.specie_collection.keys():
            g.nodes[sp]["p"] = 0
        for sp in rxn_graph.source_species:
            key = rxn_graph.specie_collection.get_key(sp)
            g.nodes[key]["p"] = 1
        # now starting the MolRank iterations
        for _ in range(n_iterations):
            # calculating rate of all reactions (cosidering new MolRank-Specie scores)
            for rxn in rxn_graph.reaction_collection.keys():
                g.nodes[rxn]["rate"] = np.prod([ajr.loc[s,"p"] for s in g.predecessors(rxn)]) * g.nodes[rxn]["obj"].properties["k"]
            # assigning total rates for seed species - needs to be after reaction rate estimation
            for sp in rxn_graph.specie_collection.keys():
                g.nodes[sp]["total_rate"] = self._calc_total_out_rate(g, sp)
            # MAIN PART: assigning MolRank ranks for species
            for sp in rxn_graph.specie_collection.keys():
                val = 0
                for rxn in g.predecessors(sp):
                    # taking the product of probability transitions
                    val += np.prod([self._calc_rate_fraction(g, rxn, sp) for sp in g.predecessors(rxn)])
                # the outcome of the influxes is the rank
                g.nodes[sp]["p"] = val
            # after specie ranking, we can rank the reactions
            for rxn in rxn_graph.reaction_collection.keys():
                ajr.loc[rxn, "p"] = max([self._calc_rate_fraction(g, rxn, sp) for sp in g.predecessors(rxn)])
        return ajr


    def apply_species(self, rxn_graph: RxnGraph) -> RxnGraph:
        # ranking all species
        df = self.rank_species(rxn_graph)
        # getting distances
        analyzer = ShortestPathAnalyzer(rxn_graph, prop_func=lambda rxn: 1)
        df["dist"] = analyzer.shortest_path_table["dist"]
        # calculating "shell normalized" MolRank
        df["rank"] = [p / sum(df[df["dist"] == dist]["p"].values) for p, dist in df[["p", "dist"]].values]
        # removing all species with rank < th
        species = df[df["rank"] < self.rank_th]["specie"].values
        res = rxn_graph.remove_species(species)
        return res

    def apply_reactions(self, rxn_graph: RxnGraph) -> RxnGraph:
        # ranking all species
        df = self.rank_reactions(rxn_graph)
        # adding relevance values
        past_relevance_criteria = lambda rxn: "molrank_reaction_relevence" in rxn.properties and not pd.isna(rxn.properties["molrank_reaction_relevence"])
        df["past_relevence"] = [past_relevance_criteria(rxn) for rxn in df["rxn"].values]
        # calculating relevence, if reaction was relevent before or is relevent according to criteria
        df["relevence"] = df["past_relevence"] | (df["p"] >= self.rank_th)
        # updating reactions with relevence info
        for rxn, relevant in df[["rxn", "relevence"]]:
            rxn.properties["molrank_reaction_relevence"] = relevant
        # removing all reactions with rank < th
        rxns = df[df["relevence"] == False]["rxn"].values
        res = rxn_graph.remove_reactions(rxns)
        return res

    def apply(self, rxn_graph: RxnGraph) -> RxnGraph:
        if self.estimate_max_constants:
            rxn_graph = assign_maximal_rate_constants(rxn_graph, self.temperature, self.energy_conversion_factor, self.specie_energy_property_name, self.rate_constant_property)
        if self.target == "species":
            return self.apply_species(rxn_graph)
        else:
            return self.apply_reactions(rxn_graph)
