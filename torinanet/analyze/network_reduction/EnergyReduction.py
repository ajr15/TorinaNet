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
                s_energy = analyzer.get_distance_from_source(specie)
                if s_energy > self.sp_energy_th:
                    rxn_graph = rxn_graph.remove_specie(specie)
        # analyzing reaction energies
        for rxn in rxn_graph.reactions:
            r_energy = self.calc_reaction_energy(rxn)
            if r_energy > self.reaction_energy_th:
                rxn_graph = rxn_graph.remove_reaction(rxn)
        return rxn_graph

class MvcEnergyReduction:

    """Method to reduce a graph based on energies of MVC species"""

    def __init__(self, reaction_energy_th: float, specie_energy_th: float, use_shortest_paths=False, sp_energy_th: float=20):
        pass

    def apply(self, rxn_graph):
        pass