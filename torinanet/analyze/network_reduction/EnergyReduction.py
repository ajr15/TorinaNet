from typing import Optional
from ..algorithms.ShortestPathAnalyzer import ShortestPathAnalyzer

class SimpleEnergyReduction: 

    """Class that implements simple reaction energy difference reduction"""

    def __init__(self, reaction_energy_th: float, use_shortest_paths: bool=False, sp_energy_th: Optional[float]=None):
        self.reaction_energy_th = reaction_energy_th
        self.sp_energy_th = sp_energy_th
        self.use_shortest_paths = use_shortest_paths

    @staticmethod
    def calc_reaction_energy(reaction):
        """Calculate energy difference between species and reactants in a reaction"""
        reactants_e = sum([s.properties["energy"] for s in reaction.reactants])
        products_e = sum([s.properties["energy"] for s in reaction.products])
        return products_e - reactants_e

    def apply(self, rxn_graph):
        # analyzing shortest paths if needed
        if self.use_shortest_paths:
            prop_func = self.calc_reaction_energy
            analyzer = ShortestPathAnalyzer(rxn_graph, prop_func=prop_func)
            for specie in rxn_graph.species:
                # ensure specie is in graph after some reductions
                if rxn_graph.has_specie(specie):
                    s_energy = analyzer.get_distance_from_source(specie)
                    if s_energy > self.sp_energy_th:
                        rxn_graph = rxn_graph.remove_specie(specie)
        # analyzing reaction energies
        for rxn in rxn_graph.reactions:
            # ensure reaction is in graph after some reductions
            if rxn_graph.has_reaction(rxn):
                r_energy = self.calc_reaction_energy(rxn)
                if r_energy > self.reaction_energy_th:
                    rxn_graph = rxn_graph.remove_reaction(rxn)
        return rxn_graph

class MinEnergyReduction:

    """Method to reduce a graph based on minimal specie energy"""

    def __init__(self, reaction_energy_th: float,
                 min_electron_energy: float,
                 use_shortest_paths=False,
                 sp_energy_th: float=20):
        self.min_electron_energy = min_electron_energy
        self.reducer = SimpleEnergyReduction(reaction_energy_th, use_shortest_paths, sp_energy_th)

    def apply(self, rxn_graph):
        # for every specie in the graph, if it doesn't have energy data - put the minimum value
        for specie in rxn_graph.species:
            if not "energy" in specie.properties:
                n_electrons = sum(specie.ac_matrix.get_atoms())
                specie.properties["energy"] = self.min_electron_energy * n_electrons
        return self.reducer.apply(rxn_graph)
