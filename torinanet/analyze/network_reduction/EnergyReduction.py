from typing import Optional
import pandas as pd
from ..algorithms.ShortestPathAnalyzer import ShortestPathAnalyzer

def calc_reaction_energy(reaction):
    """Calculate energy difference between species and reactants in a reaction"""
    reactants_e = sum([s.properties["energy"] for s in reaction.reactants])
    products_e = sum([s.properties["energy"] for s in reaction.products])
    return products_e - reactants_e


class SimpleEnergyReduction: 

    """Class that implements simple reaction energy difference reduction"""

    def __init__(self, reaction_energy_th: float, use_shortest_paths: bool=False, sp_energy_th: Optional[float]=None):
        self.reaction_energy_th = reaction_energy_th
        self.sp_energy_th = sp_energy_th
        self.use_shortest_paths = use_shortest_paths

    def apply(self, rxn_graph):
        # analyzing shortest paths if needed
        if self.use_shortest_paths:
            prop_func = lambda rxn: max(calc_reaction_energy(rxn), 0)
            analyzer = ShortestPathAnalyzer(rxn_graph, prop_func=prop_func)
            # collecting all species with too high shortest path
            ajr = []
            for specie in rxn_graph.species:
                path_rxns = analyzer.get_path_to_source(specie)
                s_energy = sum([calc_reaction_energy(rxn) for rxn in path_rxns])
                if s_energy > self.sp_energy_th:
                    ajr.append(specie)
            rxn_graph = rxn_graph.remove_species(ajr)
        # analyzing reaction energies
        ajr = []
        for rxn in rxn_graph.reactions:
            r_energy = calc_reaction_energy(rxn)
            if r_energy > self.reaction_energy_th:
                ajr.append(rxn)
        rxn_graph = rxn_graph.remove_reactions(ajr)
        return rxn_graph

class AtomicEnergyReducer:

    """Method to reduce a graph based on minimal specie atomic energy"""

    def __init__(self, reaction_energy_th: float,
                 min_atomic_energy: float,
                 use_shortest_paths=False,
                 sp_energy_th: float=20):
        self.min_atomic_energy = min_atomic_energy
        self.reducer = SimpleEnergyReduction(reaction_energy_th, use_shortest_paths, sp_energy_th)

    def apply(self, rxn_graph):
        # for every specie in the graph, if it doesn't have energy data - put the minimum value
        for specie in rxn_graph.species:
            if not "energy" in specie.properties:
                specie.properties["energy"] = self.min_atomic_energy
        return self.reducer.apply(rxn_graph)
    
class LeafEnergyReducer:

    """Apply unique thermodynamic critiria on 'leaf' reactions and species (reactions leading to specie with in_degree < min_in_degree)"""

    def __init__(self, min_in_degree: int, reaction_energy_th: float, use_shortest_paths: bool=False, sp_energy_th: Optional[float]=None):
        self.min_in_degree = min_in_degree
        self.reaction_energy_th = reaction_energy_th
        self.use_shortest_paths = use_shortest_paths
        self.sp_energy_th = sp_energy_th

    def find_leafs(self, rxn_graph):
        """Method to find leaf reactions and species"""
        # collecting all leaf reactions and species
        G = rxn_graph.to_networkx_graph(use_internal_id=True)
        leaf_reactions = []
        leaf_species = []
        for sp in rxn_graph.species:
            sid = rxn_graph.specie_collection.get_key(sp)
            generating_rxns = list(G.predecessors(sid))
            if len(generating_rxns) <= self.min_in_degree:
                leaf_reactions += [G.node[node]["obj"] for node in generating_rxns]
                leaf_species.append(sp)
        return leaf_reactions, leaf_species

    def apply(self, rxn_graph):
        leaf_reactions, leaf_species = self.find_leafs(rxn_graph)
        # first removing reactions
        ajr = []
        for rxn in leaf_reactions:
            r_energy = calc_reaction_energy(rxn)
            if r_energy > self.reaction_energy_th:
                ajr.append(rxn)
        rxn_graph = rxn_graph.remove_reactions(ajr)
        # if requested, use shortest paths to remove species
        if self.use_shortest_paths:
            prop_func = lambda rxn: max(calc_reaction_energy(rxn), 0)
            analyzer = ShortestPathAnalyzer(rxn_graph, prop_func=prop_func)
            ajr = []
            for specie in leaf_species:
                if rxn_graph.has_specie(specie):
                    path_rxns = analyzer.get_path_to_source(specie)
                    s_energy = sum([calc_reaction_energy(rxn) for rxn in path_rxns])
                    if s_energy > self.sp_energy_th:
                        ajr.append(specie)
            rxn_graph = rxn_graph.remove_species(ajr)
        return rxn_graph