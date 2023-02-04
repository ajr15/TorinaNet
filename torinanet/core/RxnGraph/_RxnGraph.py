from typing import List, Dict
import networkx as nx

from .BaseRxnGraph import BaseRxnGraph
from ..AcMatrix.BinaryAcMatrix import BinaryAcMatrix
from ..Reaction import Reaction
from ..Specie import Specie

class RxnGraph (BaseRxnGraph):
    """Default reaction graph object. Using binary ac matrix."""

    def __init__(self, use_charge=False):
        super().__init__(BinaryAcMatrix, use_charge=use_charge)
        self._species = [] # list of species in reaction graph
        self._reactions = [] # list of reactions in reaction graph
        self._specie_ids = dict() # specie ID dict to contain texts with species properties
        self._rxn_ids = dict() # reaction ID dict to contain texts with reaction properties


    @property
    def reactions(self) -> List[Reaction]:
        return self._reactions


    @property
    def species(self) -> List[Specie]:
        return self._species


    def add_specie(self, specie: Specie) -> Specie:
        s = self._read_specie_with_ac_matrix(specie)
        # check if specie is in graph
        specie_id = self.make_unique_id(s)
        if specie_id in self._specie_ids:
            return self.species[self._specie_ids[specie_id]]
        else:
            # append to graph
            self._specie_ids[specie_id] = len(self.species) # assign idx in graph for specie
            if not "visited" in s.properties:
                s.properties["visited"] = False
            self.species.append(s)
            return self.species[-1]


    def has_specie(self, specie: Specie) -> bool:
        s = self._read_specie_with_ac_matrix(specie)
        specie_id = self.make_unique_id(s)
        return specie_id in self._specie_ids


    def has_reaction(self, reaction: Reaction) -> bool:
        rxn_id = self.make_unique_id(reaction)
        return rxn_id in self._rxn_ids


    def add_reaction(self, reaction: Reaction) -> None:
        # checking if reaction is not in graph
        rxn_id = self.make_unique_id(reaction)
        if len(self.reactions) == 0 or not rxn_id in self._rxn_ids:
            ajr = Reaction(None, None, {}) # init from empty values, just like in Reaction.from_ac_matrices
            # converting reaction species by graph species (and adds new species to graph)
            for s in reaction.reactants:
                ajr.reactants.append(self.add_specie(s))
            for s in reaction.products:
                ajr.products.append(self.add_specie(s))
            # updating properties
            ajr.properties.update(reaction.properties)
            ajr._to_convension()
            # adds reaction with graph species
            self._rxn_ids[rxn_id] = len(self.reactions) # assign idx in graph for specie
            self._reactions.append(ajr)


    def compare_species(self, rxn_graph) -> List[Specie]:
        diff_species = []
        if len(self.species) < len(rxn_graph.species):
            for s in rxn_graph.species:
                s_id = self.make_unique_id(s)
                if not s_id in self._specie_ids:
                    diff_species.append(s)
        else:
            for s in self.species:
                s_id = self.make_unique_id(s)
                if not s_id in rxn_graph._specie_ids:
                    diff_species.append(s)
        return diff_species


    def has_specie_id(self, sid: str) -> bool:
        """Check if graph contains a specie ID"""
        return sid in self._specie_ids
    

    def has_reaction_id(self, rid: str) -> bool:
        """Check if graph contains a reaction ID"""
        return rid in self._rxn_ids
    

    def get_specie_from_id(self, specie_id: str) -> Specie:
        return self.species[self._specie_ids[specie_id]]


    def get_reaction_from_id(self, reaction_id: str) -> Reaction:
        return self.reactions[self._rxn_ids[reaction_id]]


    def _make_specie_index(self) -> Dict[str, int]:
        # saving redundant indexing of species
        return self._specie_ids

    def to_dict(self) -> dict:
        return {}
