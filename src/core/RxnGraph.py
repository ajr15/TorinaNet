import networkx as nx
# import dask.dataframe as dd
import numpy as np
from copy import copy
from .Reaction import Reaction

class RxnGraph:
    """Abstract Graph object. Contains all data and handling of a reaction graph"""

    def __init__(self):
        self.species = [] # list of species in reaction graph
        self.reactions = [] # list of reactions in reaction graph

    # def get_specie(self, idx):
    #     """Method to get a specie from the graph.
    #     ARGS:
    #         - idx (int): index of specie in the graph
    #     RETURNS:
    #         A Specie object"""
    #     pass

    # def get_reaction(self, idx):
    #     """Method to get a reaction from the graph.
    #     ARGS:
    #         - idx (int): index of reaction in the graph
    #     RETURNS:
    #         A Reaction object"""
    #     pass

    def add_specie(self, specie):
        """Method to add a specie to reaction graph.
        ARGS:
            - specie (Specie): Specie object
        RETURNS:
            added specie from reaction graph"""
        # check if specie is in graph
        for i in range(len(self.species)):
            if self.species[i] == specie:
                return self.species[i]
        # if not in graph check if it has an identifier
        # if specie.identifier == None:
        #     specie.identifier = str(len(self.species))
        # append to graph
        self.species.append(specie)
        return self.species[-1]

    def add_reaction(self, reaction):
        """Method to add a reaction to reaction graph.
        ARGS:
            - reaction (Reaction): Reaction object"""
        # checking if reaction is not in graph
        if len(self.reactions) == 0 or not any([s == reaction for s in self.reactions]):
            # converting reaction species by graph species (and adds new species to graph)
            reactants = []
            for s in reaction.reactants:
                reactants.append(self.add_specie(s))
            products = []
            for s in reaction.products:
                products.append(self.add_specie(s))
            # adds reaction with graph species
            self.reactions.append(Reaction(reactants, products, reaction.properties))

    def join(self, rxn_graph):
        """Method to join two reaction graphs"""
        for rxn in rxn_graph.reactions:
            self.add_reaction(rxn)

    def save_to_file(self, path):
        """Save the graph information to a file.
        ARGS:
            - path (str): path for graph files"""
        pass

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

    def compare_species(self, rxn_graph):
        """Method to get the different species between two graphs"""
        diff_species = []
        if len(self.species) < len(rxn_graph.species):
            for s in rxn_graph.species:
                if not any([s == sp  for sp in self.species]):
                    diff_species.append(s)
        else:
            for s in self.species:
                if not any([s == sp  for sp in rxn_graph.species]):
                    diff_species.append(s)
        return diff_species