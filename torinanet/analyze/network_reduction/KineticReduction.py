from typing import Dict, Optional

import numpy as np
from ...core.Specie import Specie
from ...core.RxnGraph import BaseRxnGraph
from ..kinetics.KineticAnalyzer import KineticAnalyzer
from ..kinetics.utils import assign_maximal_rate_constants

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
        
    
    def apply(self, rxn_graph: BaseRxnGraph):
        # if required, estimate maximal rate constants
        if self.estimate_max_constants:
            rxn_graph = assign_maximal_rate_constants(rxn_graph, self.temperature, self.energy_conversion_factor, self.specie_energy_property_name, self.rate_constant_property)
        if not self.reactant_concs:
            self.reactant_concs = {s: 1 for s in rxn_graph.source_species}
        # initializing kinetic solver
        analyzer = KineticAnalyzer(rxn_graph, self.rate_constant_property)
        # building initial concentrations
        initial_concs = np.zeros(len(rxn_graph.species))
        for s, v in self.reactant_concs:
            initial_concs[analyzer.get_specie_index(s)] = v
        # solving kinetics
        analyzer.solve_kinetics(self.simulation_time, self.timestep, initial_concs, **self.solver_kwargs)
        # analyzing reaction rates
        for rxn in rxn_graph.reactions:
            # ensure reaction is in graph after some reductions
            if rxn_graph.has_reaction(rxn):
                rate = analyzer.get_max_rate(rxn)
                if rate > self.reaction_rate_th:
                    rxn_graph = rxn_graph.remove_reaction(rxn)
        return rxn_graph
