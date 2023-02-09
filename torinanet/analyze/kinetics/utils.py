import numpy as np
from ...core.RxnGraph import RxnGraph

def assign_maximal_rate_constants(rxn_graph: RxnGraph, temperature: float, energy_conversion_factor: float=4.359744e-18, specie_energy_property_name: str="energy", rate_constant_property_name: str="k") -> RxnGraph:
    """Method to assign the largest possible rate constant to a reaction (minimal activation energy), based on the eyring-polanyi equation"""
    kb = 1.3806e-23
    h = 6.62607e-34
    for rxn in rxn_graph.reactions:
        dE = 0
        for s in rxn.reactants:
            if specie_energy_property_name in s.properties:
                dE = dE - s.properties[specie_energy_property_name]
            else:
                raise RuntimeError("All species in network must have \'{}\' property to estimate rate constants".format(specie_energy_property_name))
        for s in rxn.products:
            if specie_energy_property_name in s.properties:
                dE = dE + s.properties[specie_energy_property_name]
            else:
                raise RuntimeError("All species in network must have \'{}\' property to estimate rate constants".format(specie_energy_property_name))
        # fixing dE value to 0 for exothermic reactions
        dE = dE if dE > 0 else 0
        # sets rate constant
        rxn.properties[rate_constant_property_name] = kb * temperature / h * np.exp(- (dE * energy_conversion_factor) / (kb * temperature))
    return rxn_graph