from typing import List
import networkx as nx
from networkx.classes.function import neighbors
# import dask.dataframe as dd
import numpy as np
from copy import copy
import os

from .AcMatrix.BinaryAcMatrix import BinaryAcMatrix
from .Reaction import Reaction
from .Specie import Specie
from ..utils.TimeFunc import TimeFunc

class RxnGraph:
    """Reaction Graph object. Contains all data and handling of a reaction graph"""

    def __init__(self):
        self.reactant_species = None # list of reactants species (origin of the graphs)
        self.species = [] # list of species in reaction graph
        self.reactions = [] # list of reactions in reaction graph
        self._specie_ids = dict() # specie ID dict to contain texts with species properties
        self._rxn_ids = dict() # reaction ID dict to contain texts with reaction properties

    def set_reactant_species(self, species, force_adding=False):
        # check if species are in graph and using graph species
        reactants = []
        for specie in species:
            specie_id = specie._get_id_str()
            if not specie_id in self._specie_ids:
                if not force_adding:
                    raise ValueError("Some of the required species ({}) are not in the graph".format(specie.identifier))
                else:
                    s = self.add_specie(specie)
                    reactants.append(s)
            else:
                reactants.append(self.species[self._specie_ids[specie_id]])
        # set reactant species
        self.reactant_species = reactants

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

    def has_specie(self, specie):
        specie_id = specie._get_id_str()
        return specie_id in self._specie_ids

    def has_reaction(self, reaction):
        rxn_id = reaction._get_id_str()
        return rxn_id in self._rxn_ids

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

    @staticmethod
    def _dfs_remove_id_str(network: nx.DiGraph, id_str: str, sources: List[str]):
        """Internal ethod to remove an object with given id_str from the reaction graph. Returns list of node id strings to keep in graph.
        ARGS:
            - id_str (str): id string for the desired object to remove (reaction or specie)
            - sources (List[str]): list of graph sources
        RETURNS:
            (List[str]) list of id strings of nodes in reaction graph to keep"""
        # converting to networkx DiGraph object and removing desired node
        network.remove_node(id_str)
        # making set with nodes id strings to remove
        res = set()
        for source in sources:
            if source._get_id_str() in network:
                res = res.union(set(nx.algorithms.dfs_tree(network, source._get_id_str()).nodes))
        return res

    def copy(self, keep_ids=None):
        """Method to copy the reaction graph.
        ARGS:
            - keep_ids (List[str] or None): optional to copy the reaction graph only with the given reaction id strings. default=None
        RETURNS:
            (RxnGraph) a copy of the reaction graph"""
        # making list of reaction idxs for copying
        if keep_ids is not None:
            rxn_idxs = []
            for id_str in keep_ids:
                if id_str in self._rxn_ids:
                    rxn_idxs.append(self._rxn_ids[id_str])
        else:
            rxn_idxs = self._rxn_ids.values()
        # copying reaction graph
        nrxn_graph = RxnGraph()
        for idx in rxn_idxs:
            nrxn_graph.add_reaction(self.reactions[idx])
        # adding reactant species
        nrxn_graph.set_reactant_species(self.reactant_species)
        return nrxn_graph
        
    def remove_specie(self, specie: Specie):
        """Method to remove a specie from the reaction graph.
        ARGS:
            - specie (Specie): desired specie to remove
        RETURNS:
            (RxnGraph) a copy of the reaction graph without the speice"""
        # check if specie is in graph
        specie_id = specie._get_id_str()
        if not self.has_specie(specie):
            raise ValueError("Desired specie is not in the graph !")
        # check if there are defined reactants in the graph
        if self.reactant_species is None:
            raise NotImplementedError("The graph doesnt have defined reactants. This method is undefined in this case")
        # convert to networkx object
        network = self.to_networkx_graph(use_internal_id=True)
        # removing reactions that the specie is a reactant in
        rxns_to_remove = set([rxn for rxn in nx.all_neighbors(network, specie_id)])
        network.remove_nodes_from(rxns_to_remove)
        # running dfs to find other nodes to remove
        keep_ids = self._dfs_remove_id_str(network, specie_id, self.reactant_species)
        # building a copy of RxnGraph without specie
        nrxn_graph = RxnGraph()
        for rxn_id in self._rxn_ids:
            if rxn_id in keep_ids:
                rxn = self.reactions[self._rxn_ids[rxn_id]]
                if rxn._get_id_str() in rxns_to_remove:
                    continue
                else:
                    nrxn_graph.add_reaction(rxn)
        # making sure nrxn has the same reactant species as parent graph
        nrxn_graph.set_reactant_species(self.reactant_species, force_adding=True)
        # returning a copy of the network with the required ids
        return nrxn_graph

    def remove_reaction(self, reaction: Reaction):
        """Method to remove a reaction from the reaction graph.
        ARGS:
            - reaction (Reaction): desired reaction to remove
        RETURNS:
            (RxnGraph) a copy of the reaction graph without the speice"""
        # check if specie is in graph
        reaction_id = reaction._get_id_str()
        if not reaction_id in self._rxn_ids:
            raise ValueError("Desired reaction is not in the graph !")
        # check if there are defined reactants in the graph
        if self.reactant_species is None:
            raise NotImplementedError("The graph doesnt have defined reactants. This method is undefined in this case")
        # convert to networkx object
        network = self.to_networkx_graph(use_internal_id=True)
        # running dfs to find other nodes to remove
        keep_ids = self._dfs_remove_id_str(network, reaction_id, self.reactant_species)
        # building a copy of RxnGraph without specie
        nrxn_graph = RxnGraph()
        for rxn_id in self._rxn_ids:
            if rxn_id in keep_ids:
                rxn = self.reactions[self._rxn_ids[rxn_id]]
                nrxn_graph.add_reaction(rxn)
        # making sure nrxn has the same reactant species as parent graph
        nrxn_graph.set_reactant_species(self.reactant_species)
        # returning a copy of the network with the required ids
        return nrxn_graph

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
                smiles = s.ac_matrix.to_specie().identifier
                f.write(smiles)
            f.write("****\n")
            for r in self.reactions:
                st = ""
                for sp in r.reactants:
                    st += str(self._specie_ids[sp._get_id_str()]) + ","
                st += "="
                for sp in r.products:
                    st += str(self._specie_ids[sp._get_id_str()]) + ","
                st += "\n"
                f.write(st)

    def from_file(self, path):
        readSpecies = True
        readReactions = False
        with open(path, "r") as f:
            for line in f.readlines():
                if "****" in line:
                    readSpecies = False
                    readReactions = True
                elif readSpecies:
                    s = BinaryAcMatrix.from_specie(Specie(line)).to_specie()
                    s.identifier = line.strip()
                    self.add_specie(s)
                elif readReactions:
                    reactant_idxs = [int(i) for i in line.split("=")[0].split(",") if not len(i) == 0]
                    reactants = [self.species[i] for i in reactant_idxs]
                    product_idxs = [int(i) for i in line.split("=")[1].split(",") if not len(i) == 0 and not i == '\n']
                    products = [self.species[i] for i in product_idxs]
                    rxn = Reaction(reactants, products)
                    self.add_reaction(rxn)

    def project_to_species(self):
        """Method to convert a bipartite reaction graph to a monopartite species graph"""
        pass

    def project_to_reactions(self):
        """Method to convert a bipartite reaction graph to a monopartite reactions graph"""
        pass

    def to_networkx_graph(self, use_internal_id=False):
        """Convert RxnGraph object to a bipartite networkx graph.
        ARGS:
            - use_internal_id (bool): weather to use the internal id for reactions and species, for more robust conversion. default=False"""
        # adds identifiers for species
        if not use_internal_id:
            for i in range(len(self.species)):
                if not type(self.species[i].identifier) is str:
                    self.species[i].identifier = str(i)
            network = nx.DiGraph()
            for i, rxn in enumerate(self.reactions):
                for s in rxn.reactants:
                    network.add_edge(str(s.identifier), int(i))
                for s in rxn.products:
                    network.add_edge(int(i), str(s.identifier))
        else:
            network = nx.DiGraph()
            for i, rxn in enumerate(self.reactions):
                rxn_id = rxn._get_id_str()
                for s in rxn.reactants:
                    network.add_edge(s._get_id_str(), rxn_id)
                for s in rxn.products:
                    network.add_edge(rxn_id, s._get_id_str())
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