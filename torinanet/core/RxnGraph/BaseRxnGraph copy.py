from abc import abstractclassmethod, ABC, abstractproperty
from typing import List, Set
import os
import pandas as pd
import networkx as nx
import json
from zipfile import ZipFile
from io import StringIO
from .. import RxnGraph
from ..Specie import Specie
from ..Reaction import Reaction
from ..AcMatrix.AcMatrix import AcMatrix

class BaseRxnGraph (ABC):

    """Abstract reaction graph object
    ARGS:
        - ac_matrix_type (TorinaNet.AcMatrix): type of AC matrix for specie representation to be used in the graph
    ATTRIBUTES:
        ac_matrix_type (TorinaNet.AcMatrix): type of AC matrix for specie representation to be used in the graph
        source_species (List[Specie]): list of source species (reactants) for the reaction graph"""

    def __init__(self, ac_matrix_type: AcMatrix, use_charge=False) -> None:
        self.ac_matrix_type = ac_matrix_type
        self.source_species = None
        self.use_charge = use_charge
    
    # def make_unique_id(self, obj):
    #     return obj._get_charged_id_str() if self.use_charge else obj._get_id_str()

    @abstractproperty
    def reactions(self) -> List[Reaction]:
        """List of all reactions in reaction graph"""
        pass

    
    @reactions.setter
    def reactions(self, *args):
        raise RuntimeError("Cannot set reactions for reaction graph")
    
    
    @reactions.getter
    def reactions(self):
        pass

    @abstractproperty
    def species(self) -> List[Specie]:
        """List of all species in reaction graph"""
        pass

    
    @species.setter
    def species(self, *args):
        raise RuntimeError("Cannot set species for reaction graph")
    
    
    @species.getter
    def species(self):
        pass


    def _read_specie_with_ac_matrix(self, specie: Specie) -> Specie:
        """Method to add an AC matrix to a specie. CRUCIAL for graph performance.
        ARGS:
            - specie (Specie): specie to add AC matrix for
        RETURNS:
            (Specie) copy of the specie with the required AC matrix"""
        if specie.ac_matrix is None:
            s = Specie.from_ac_matrix(self.ac_matrix_type.from_specie(specie))
            s.properties.update(specie.properties)
            s.charge = specie.charge
            return s
        else:
            return specie

    
    def set_source_species(self, species: List[Specie], force=False) -> None:
        """Method to set source species for the reaction graph.
        ARGS:
            - species (List[Specie]): list of species to add to graph
            - force (bool): add species also if they are not in graph (default=False)"""
        source = []
        for specie in species:
            if not self.has_specie(specie):
                if not force:
                    raise ValueError("Some of the required species ({}) are not in the graph".format(specie.identifier))
                else:
                    s = self.add_specie(specie)
                    source.append(s)
            else:
                source.append(self.add_specie(specie))
        # set reactant species
        self.source_species = source


    @abstractclassmethod
    def add_specie(self, specie: Specie) -> Specie:
        """Method to add a specie to reaction graph.
        ARGS:
            - specie (Specie): Specie object
        RETURNS:
            added specie from reaction graph"""
        pass


    @abstractclassmethod
    def add_reaction(self, reaction: Reaction) -> None:
        """Method to add a reaction to reaction graph.
        ARGS:
            - reaction (Reaction): Reaction object"""
        pass


    def to_networkx_graph(self, use_internal_id=False) -> nx.DiGraph:
        """Convert RxnGraph object to a bipartite networkx graph. Defaults for using the provided specie identifier and reaction index as node names.
        ARGS:
            - use_internal_id (bool): weather to use the internal id for reactions and species, for more robust conversion. default=False
        RETURNS:
            (nx.DiGraph) networkx bipartite graph representing the reaction graph"""
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
                rxn_id = self.make_unique_id(rxn)
                for s in rxn.reactants:
                    network.add_edge(self.make_unique_id(s), rxn_id)
                for s in rxn.products:
                    network.add_edge(rxn_id, self.make_unique_id(s))
        return network

    @abstractclassmethod
    def compare_species(self, rxn_graph) -> List[Specie]:
        """Method to get the different species between two reaction graphs
        ARGS:
            - rxn_graph (TorinaNet.RxnGraph): the reaction graph to compare to
        RETURNS:
            (List[Specie]) list of different species between graphs"""
        pass

    @abstractclassmethod
    def has_specie(self, specie: Specie) -> bool:
        """Method to check if reaction graph has a specie.
        ARGS:
            - specie (TorinaNet.Specie): specie to check
        RETURNS:
            (bool) wheather specie is in graph"""
        raise NotImplementedError("has_specie method is not implemented for this graph")

    @abstractclassmethod
    def has_reaction(self, reaction: Reaction) -> bool:
        """Method to check if reaction graph has a reaction.
        ARGS:
            - reaction (TorinaNet.Reaction): reaction to check
        RETURNS:
            (bool) wheather reaction is in graph"""
        raise NotImplementedError("has_reaction method is not implemented for this graph")

    @abstractclassmethod
    def get_specie_from_id(self, specie_id: str) -> Specie:
        """Method to get specie object from reaction graph, given specie id.
        ARGS:
            - specie_id (str): specie id generated from Specie._gen_id_str
        RETURNS:
            (Specie) required specie object from reaction graph or ValueError if specie is not found"""
        raise NotImplementedError("get_specie_from_id is not implemented for this subclass")

    @abstractclassmethod
    def get_reaction_from_id(self, specie_id: str) -> Reaction:
            """Method to get reaction object from reaction graph, given reaction id.
            ARGS:
                - reaction_id (str): specie id generated from Reaction._gen_id_str
            RETURNS:
                (Reaction) required reaction object from reaction graph or ValueError if reaction is not found"""
            raise NotImplementedError("get_reaction_from_id is not implemented for this subclass")


    def _dfs_remove_id_str(self, network: nx.DiGraph, id_str: str, sources: List[Specie]) -> Set[str]:
        """Internal ethod to remove an object with given id_str from the reaction graph. Returns list of node id strings to keep in graph.
        ARGS:
            - id_str (str): id string for the desired object to remove (reaction or specie)
            - sources (List[str]): list of graph sources
        RETURNS:
            (Set[str]) set of id strings of nodes in reaction graph to keep"""
        # converting to networkx DiGraph object and removing desired node
        network.remove_node(id_str)
        # making set with nodes id strings to remove
        res = set()
        for source in sources:
            if self.make_unique_id(source) in network:
                res = res.union(set(nx.algorithms.dfs_tree(network, self.make_unique_id(source)).nodes))
        return res

    def copy(self, keep_ids=None, use_charge=False):
        """Method to copy the reaction graph.
        ARGS:
            - keep_ids (List[str] or None): optional to copy the reaction graph only with the given reaction id strings. default=None
        RETURNS:
            (RxnGraph) a copy of the reaction graph"""
        # making list of reaction idxs for copying
        if keep_ids is not None:
            rxns = []
            for id_str in keep_ids:
                rxns.append(self.get_reaction_from_id(id_str))
        else:
            rxns = self.reactions
        # copying reaction graph
        nrxn_graph = self.__class__(use_charge=use_charge)
        for rxn in rxns:
            nrxn_graph.add_reaction(rxn)
        # adding reactant species
        nrxn_graph.set_source_species(self.source_species, force=True)
        return nrxn_graph
        
    def remove_specie(self, specie: Specie):
        """Method to remove a specie from the reaction graph.
        ARGS:
            - specie (Specie): desired specie to remove
        RETURNS:
            (RxnGraph) a copy of the reaction graph without the speice"""
        # check if specie is in graph
        if not self.has_specie(specie):
            raise ValueError("Desired specie is not in the graph !")
        # check if there are defined reactants in the graph
        if self.source_species is None:
            raise NotImplementedError("The graph doesnt have defined source reactants. This method is undefined in this case")
        specie_id = self.make_unique_id(specie)
        # convert to networkx object
        network = self.to_networkx_graph(use_internal_id=True)
        # removing reactions that the specie is a reactant in
        rxns_to_remove = set([rxn for rxn in nx.all_neighbors(network, specie_id)])
        network.remove_nodes_from(rxns_to_remove)
        # running dfs to find other nodes to remove
        keep_ids = self._dfs_remove_id_str(network, specie_id, self.source_species)
        # building a copy of RxnGraph without specie
        nrxn_graph = self.__class__(use_charge=self.use_charge)
        for rxn_id in keep_ids:
            if "=" in rxn_id: # keep_ids has also specie ids in it... this is filtering it. TODO: make it better
                if rxn_id in rxns_to_remove:
                    continue
                else:
                    rxn = self.get_reaction_from_id(rxn_id)
                    nrxn_graph.add_reaction(rxn)
        # making sure nrxn has the same reactant species as parent graph
        nrxn_graph.set_source_species(self.source_species, force=True)
        # returning a copy of the network with the required ids
        return nrxn_graph

    def remove_reaction(self, reaction: Reaction):
        """Method to remove a reaction from the reaction graph.
        ARGS:
            - reaction (Reaction): desired reaction to remove
        RETURNS:
            (RxnGraph) a copy of the reaction graph without the speice"""
        # check if specie is in graph
        if not self.has_reaction(reaction):
            raise ValueError("Desired reaction is not in the graph !")
        # check if there are defined reactants in the graph
        if self.source_species is None:
            raise NotImplementedError("The graph doesnt have defined source reactants. This method is undefined in this case")
        reaction_id = self.make_unique_id(reaction)
        # convert to networkx object
        network = self.to_networkx_graph(use_internal_id=True)
        # running dfs to find other nodes to remove
        keep_ids = self._dfs_remove_id_str(network, reaction_id, self.source_species)
        # building a copy of RxnGraph without specie
        nrxn_graph = self.__class__(use_charge=self.use_charge)
        for rxn_id in keep_ids:
            if "=" in rxn_id: # same problem as in remove_specie..
                rxn = self.get_reaction_from_id(rxn_id)
                nrxn_graph.add_reaction(rxn)
        # making sure nrxn has the same reactant species as parent graph
        nrxn_graph.set_source_species(self.source_species, force=True)
        # returning a copy of the network with the required ids
        return nrxn_graph

    def join(self, rxn_graph) -> None:
        """Method to join a reaction graph into the current reaction graph.
        ARGS:
            - rxn_graph (BaseRxnGraph): reaction graph to join with (adds its reactions to current graph)"""
        for rxn in rxn_graph.reactions:
            self.add_reaction(rxn)

    def _make_specie_df(self) -> pd.DataFrame:
        """Method to convert all specie data into a pd.DataFrame object. to be used for saving / reading reaction graph files"""
        res = pd.DataFrame()
        for i, s in enumerate(self.species):
            d = dict()
            d["idx"] = i
            d["ac_mat_str"] = s.ac_matrix._to_str()
            d["sid"] = self.make_unique_id(s)
            d["identifier"] = s.identifier
            d["charge"] = s.charge
            d.update(s.properties)
            res = res.append(d, ignore_index=True)
        res = res.set_index("sid")
        return res

    def _make_reactions_df(self, species_df) -> pd.DataFrame:
        """Method to convert all reaction information into pd.DataFrame object. to be used for saving / reading reaction graph files"""
        res = pd.DataFrame()
        for r in self.reactions:
            d = {}
            d.update(r.properties)
            st = ",".join([str(int(species_df.loc[self.make_unique_id(sp), "idx"])) for sp in r.reactants]) + \
                    "=" + \
                    ",".join([str(int(species_df.loc[self.make_unique_id(sp), "idx"])) for sp in r.products])
            d["r_str"] = st
            res = res.append(d, ignore_index=True)
        if len(self.reactions) > 0:
            res = res.set_index("r_str")
        return res

    def save(self, path) -> None:
        """Save the graph information to a file.
        ARGS:
            - path (str): path to file """
        path_basename = "rxn_graph"
        # saving specie_df temporarily
        specie_df = self._make_specie_df()
        specie_df.to_csv(path_basename + "_species")
        # saving reaction_df temporarily
        rxn_df = self._make_reactions_df(specie_df)
        rxn_df.to_csv(path_basename + "_reactions")
        # saving source specie unique IDs in a temp file
        with open(path_basename + "_params", "w") as f:
            d = {
                    "charged": self.use_charge, 
                    "source_species_ids": [self.make_unique_id(s) for s in self.source_species],
                    "parent_object": type(self).__name__
                }
            json.dump(d, f)
        # saving reaction graph as archive file
        with ZipFile(path, "w") as f:
            f.write(path_basename + "_species")
            f.write(path_basename + "_reactions")
            f.write(path_basename + "_params")
        # removing temporary files
        os.remove(path_basename + "_species")
        os.remove(path_basename + "_reactions")
        os.remove(path_basename + "_params")


    @classmethod
    def from_file(cls, path):
        """Load reaction graph data from file.
        ARGS:
            - path (str): path to the reaction graph file
        RETURNS:
            (RxnGraph) reaction graph object"""
        with ZipFile(path, "r") as zipfile:
            # init
            path_basename = "rxn_graph"
            # setting graph with proper charge reading
            with zipfile.open(path_basename + "_params") as f:
                d = json.load(f)
                rxn_graph_class = getattr(RxnGraph, d["parent_object"])
                rxn_graph = rxn_graph_class(d["charged"])
            # reading specie csv from zip to pd.DataFrame
            species_df = pd.read_csv(zipfile.open(path_basename + "_species"))
            for specie_row in species_df.to_dict(orient="records"):
                # reading specie from ac_matrix string
                specie_mat = rxn_graph.ac_matrix_type()
                specie_mat._from_str(specie_row["ac_mat_str"])
                specie = specie_mat.to_specie()
                # reading charge
                specie.charge = specie_row["charge"]
                # adding properties
                for c in specie_row.keys():
                    if not c in ["idx", "ac_mat_str", "charge", "sid"]:
                        specie.properties[c] = specie_row[c]
                # adding specie to graph
                rxn_graph.add_specie(specie)
            # setting source species
            species = []
            for sid in d["source_species_ids"]:
                species.append(rxn_graph.get_specie_from_id(sid))
            rxn_graph.set_source_species(species, force=True)
            # reading reactions csv from zip to pd.DataFrame
            reactions_df = pd.read_csv(zipfile.open(path_basename + "_reactions"))
            for rxn_row in reactions_df.to_dict(orient="records"):
                # reading reaction string
                r_str = rxn_row["r_str"]
                reactant_idxs = [int(s) for s in r_str.split("=")[0].split(",") if not len(s) == 0]
                reactants = [rxn_graph.species[i] for i in reactant_idxs]
                product_idxs = [int(s) for s in r_str.split("=")[-1].split(",") if not len(s) == 0]
                products = [rxn_graph.species[i] for i in product_idxs]
                rxn = Reaction(reactants, products)
                # adding properties
                for c in rxn_row.keys():
                    if not c in ["r_str"]:
                        rxn.properties[c] = rxn_row[c]
                # pushing to graph
                rxn_graph.add_reaction(rxn)
        return rxn_graph

    def make_specie_visited(self, specie: Specie):
        self.get_specie_from_id(self.make_unique_id(specie)).properties["visited"] = True


    def get_unvisited_species(self) -> List[Specie]:
        """Method to get all unvisited species in reaction graph"""
        return [s for s in self.species if not s.properties["visited"]]


    def get_visited_species(self) -> List[Specie]:
        """Method to get all unvisited species in reaction graph"""
        return [s for s in self.species if s.properties["visited"]]

    @abstractclassmethod
    def to_dict(self) -> dict:
        """Method to return the init parameters of the object as a dictionary"""
        return {}


    def get_n_species(self) -> int:
        """Method to get the number of species in graph"""
        return len(self.species)


    def get_n_reactions(self) -> int:
            """Method to get the number of reactions in graph"""
            return len(self.reactions)

    def update_specie(self, specie: Specie):
        """Method to update existing specie entry with a different one"""
        raise NotImplementedError("Cannot update specie's properties in this reaction graph type")

    def update_reaction(self, reaction: Reaction):
        """Method to update exisiting reaction entry with a different one"""
        raise NotImplementedError("Cannot update reaction's properties in this reaction graph type")

    def is_equal(self, x):
        """Check if graph is equal to another"""
        if not isinstance(x, BaseRxnGraph):
            raise ValueError("Can compare only 2 reaction graphs")
        net1 = self.to_networkx_graph(use_internal_id=True)
        net2 = x.to_networkx_graph(use_internal_id=True)
        # using the networkx comparision mechanism
        return net1.edges == net2.edges and net1.nodes == net2.nodes

    def __eq__(self, x):
        if not isinstance(x, BaseRxnGraph):
            raise ValueError("Can compare only 2 reaction graphs")
        net1 = self.to_networkx_graph(use_internal_id=True)
        net2 = x.to_networkx_graph(use_internal_id=True)
        # using the networkx comparision mechanism
        return net1.edges == net2.edges and net1.nodes == net2.nodes
