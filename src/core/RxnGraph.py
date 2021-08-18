import networkx as nx
# import dask.dataframe as dd
import numpy as np
from copy import copy
import os
from .Reaction import Reaction
from .Specie import Specie
from ..utils.TimeFunc import TimeFunc

class RxnGraph:
    """Abstract Graph object. Contains all data and handling of a reaction graph"""

    def __init__(self):
        self.species = [] # list of species in reaction graph
        self.reactions = [] # list of reactions in reaction graph
        self._specie_ids = dict() # specie ID dict to contain texts with species properties
        self._rxn_ids = dict() # reaction ID dict to contain texts with reaction properties

    @TimeFunc
    def add_specie(self, specie):
        """Method to add a specie to reaction graph.
        ARGS:
            - specie (Specie): Specie object
        RETURNS:
            added specie from reaction graph"""
        # check if specie is in graph
        specie_id = specie._get_id_str()
        if specie_id in self._specie_ids:
            return self.species[self._specie_ids[specie_id]]
        else:
            # append to graph
            self._specie_ids[specie_id] = len(self.species) # assign idx in graph for specie
            self.species.append(specie)
            return self.species[-1]

    @TimeFunc
    def add_reaction(self, reaction):
        """Method to add a reaction to reaction graph.
        ARGS:
            - reaction (Reaction): Reaction object"""
        # checking if reaction is not in graph
        rxn_id = reaction._get_id_str()
        if len(self.reactions) == 0 or not rxn_id in self._rxn_ids:
            # converting reaction species by graph species (and adds new species to graph)
            reactants = []
            for s in reaction.reactants:
                reactants.append(self.add_specie(s))
            products = []
            for s in reaction.products:
                products.append(self.add_specie(s))
            # adds reaction with graph species
            self._rxn_ids[rxn_id] = len(self.reactions) # assign idx in graph for specie
            self.reactions.append(Reaction(reactants, products, reaction.properties))

    def join(self, rxn_graph):
        """Method to join two reaction graphs"""
        for rxn in rxn_graph.reactions:
            self.add_reaction(rxn)

    def save_to_file(self, path):
        """Save the graph information to a file.
        ARGS:
            - path (str): path to directory for graph file"""
        with open(path, "w") as f:
            for s in self.species:
                smiles = s.ac_matrix.to_specie().identifier + "\n"
                f.write(smiles)
            f.write("****\n")
            for r in self.reactions:
                st = ""
                for sp in r.reactants:
                    st += str(self._rxn_ids[sp._get_id_str()]) + ","
                st += "="
                for sp in r.products:
                    st += str(self._rxn_ids[sp._get_id_str()]) + ","
                st += "\n"
                f.write(st)

    def from_file(self, path):
        g = RxnGraph()
        readSpecies = True
        readReactions = False
        with open(path, "r") as f:
            for line in f.readlines():
                if "****" in line:
                    readSpecies = False
                    readReactions = True
                elif readSpecies:
                    g.species.append(Specie(line))
                elif readReactions:
                    rxn = Reaction()
                    reactant_idxs = [int(i) for i in line.split("=")[0].split(",")]
                    rxn.reactants = [g.species[i] for i in reactant_idxs]
                    product_idxs = [int(i) for i in line.split("=")[1].split(",")]
                    rxn.reactants = [g.species[i] for i in product_idxs]
                    g.reactions.append(rxn)

    def project_to_species(self):
        """Method to convert a bipartite reaction graph to a monopartite species graph"""
        pass

    def project_to_reactions(self):
        """Method to convert a bipartite reaction graph to a monopartite reactions graph"""
        pass

    def to_netwokx_graph(self):
        """Convert RxnGraph object to a bipartite networkx graph"""
        # adds identifiers for species
        for i in range(len(self.species)):
            if not type(self.species[i].identifier) is str:
                self.species[i].identifier = i
        network = nx.DiGraph()
        for i, rxn in enumerate(self.reactions):
            for s in rxn.reactants:
                network.add_edge(str(s.identifier), int(i))
            for s in rxn.products:
                network.add_edge(int(i), str(s.identifier))
        return network

    @TimeFunc
    def compare_species(self, rxn_graph):
        """Method to get the different species between two graphs"""
        diff_species = []
        if len(self.species) < len(rxn_graph.species):
            for s in rxn_graph.species:
                s_id = s._get_id_str()
                if not s_id in self._specie_ids:
                    diff_species.append(s)
        else:
            for s in self.species:
                s_id = s._get_id_str()
                if not s_id in rxn_graph._specie_ids:
                    diff_species.append(s)
        return diff_species