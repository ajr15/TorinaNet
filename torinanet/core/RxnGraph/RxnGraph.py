from typing import List, Optional, Generator, Union, Iterable
import os
import networkx as nx
import pandas as pd
import pickle
from zipfile import ZipFile
from .. import RxnGraph
from ..Specie import Specie
from ..Reaction import Reaction
from ..AcMatrix.AcMatrix import AcMatrix
from ..AcMatrix.BinaryAcMatrix import BinaryAcMatrix
from .HashedCollection.HashedCollection import HashedCollection
from .HashedCollection.CuckooHashedCollection import CuckooSpecieCollection, IndepCuckooReactionCollection
from ...utils.TimeFunc import TimeFunc

class RxnGraph:

    """Abstract reaction graph object
    ARGS:
        - ac_matrix_type (TorinaNet.AcMatrix): type of AC matrix for specie representation to be used in the graph
        - specie_collection (HashedCollection): type of specie collection to handle the species in the graph
        - reaction_collection (HashedCollection): type of reaction collection to handle the reactions in the graph
        - specie_collection_kwargs (Optional[dict]): keyword argument dictionary for the specie collection
        - reaction_collection_kwargs (Optional[dict]): keyword argument dictionary for the reaction collection"""

    def __init__(self, ac_matrix_type: AcMatrix=BinaryAcMatrix, specie_collection: HashedCollection=CuckooSpecieCollection, reaction_collection: HashedCollection=IndepCuckooReactionCollection, 
                    specie_collection_kwargs: Optional[dict]=None, reaction_collection_kwargs: Optional[dict]=None) -> None:
        self.ac_matrix_type = ac_matrix_type
        self.source_species = None
        self._has_charge = None
        # parsing collection arguments
        if not issubclass(specie_collection, HashedCollection):
            raise ValueError("Specie collection type must be HashedCollection")
        sp_defaults = {}
        if not specie_collection_kwargs is None:
            sp_defaults.update(specie_collection_kwargs)
        self.specie_collection = specie_collection(**sp_defaults)
        if not issubclass(reaction_collection, HashedCollection):
            raise ValueError("Reaction collection type must be HashedCollection")
        sp_defaults = {}
        if not reaction_collection_kwargs is None:
            sp_defaults.update(reaction_collection_kwargs)
        self.reaction_collection = reaction_collection(**sp_defaults)

    @property
    def reactions(self) -> Iterable[Reaction]:
        """List of all reactions in reaction graph"""
        return self.reaction_collection.objects()

    @reactions.setter
    def reactions(self, *args):
        raise RuntimeError("Cannot set reactions for reaction graph")
   
    @reactions.getter
    def reactions(self):
        return self.reaction_collection.objects()

    @property
    def species(self) -> Iterable[Specie]:
        """List of all species in reaction graph"""
        return self.specie_collection.objects()
    
    @species.setter
    def species(self, *args):
        raise RuntimeError("Cannot set species for reaction graph")
    
    @species.getter
    def species(self):
        return self.specie_collection.objects()

    @TimeFunc
    def add_specie(self, specie: Specie) -> Specie:
        """Method to add a specie to reaction graph.
        ARGS:
            - specie (Specie): Specie object
        RETURNS:
            added specie from reaction graph"""
        if self._has_charge is None:
            self._has_charge = specie.charge is not None
        elif self._has_charge is True and specie.charge is None:
            raise RuntimeError("Cannot add specie without charge information to graph with species with charge information")
        elif self._has_charge is False and specie.charge is not None:
            raise RuntimeError("Cannot add specie with charge information to graph with species without charge information")
        s = self._read_specie_with_ac_matrix(specie)
        # check if specie is in graph
        if not self.specie_collection.has(s):
            # make specie not visited if property is not set
            if not "visited" in s.properties:
                s.properties["visited"] = False
            # add specie to collection
            self.specie_collection.add(s)
        # return specie
        return self.specie_collection.get(s)

    @TimeFunc
    def add_reaction(self, reaction: Reaction) -> None:
        """Method to add a reaction to reaction graph.
        ARGS:
            - reaction (Reaction): Reaction object"""
        # only reaction only if its non-empty and not in graph
        if not self.reaction_collection.has(reaction) and len(reaction.reactants) > 0 and len(reaction.products) > 0:
            ajr = Reaction(None, None, {}) # init from empty values, just like in Reaction.from_ac_matrices
            # converting reaction species by graph species (and adds new species to graph)
            for s in reaction.reactants:
                ajr.reactants.append(self.add_specie(s))
            for s in reaction.products:
                ajr.products.append(self.add_specie(s))
            # updating properties
            ajr.properties.update(reaction.properties)
            # adds reaction with graph species
            self.reaction_collection.add(ajr)

    @TimeFunc
    def compare_species(self, rxn_graph) -> List[Specie]:
        """Method to get the different species between two reaction graphs
        ARGS:
            - rxn_graph (TorinaNet.RxnGraph): the reaction graph to compare to
        RETURNS:
            (List[Specie]) list of different species between graphs"""
        if not isinstance(rxn_graph, RxnGraph):
            raise TypeError("compare_species is defined only between RxnGraph objects")
        return list(self.specie_collection.substract(rxn_graph.specie_collection))

    @TimeFunc
    def has_specie(self, specie: Specie) -> bool:
        """Method to check if reaction graph has a specie.
        ARGS:
            - specie (TorinaNet.Specie): specie to check
        RETURNS:
            (bool) wheather specie is in graph"""
        return self.specie_collection.has(self._read_specie_with_ac_matrix(specie))

    @TimeFunc
    def has_reaction(self, reaction: Reaction) -> bool:
        """Method to check if reaction graph has a reaction.
        ARGS:
            - reaction (TorinaNet.Reaction): reaction to check
        RETURNS:
            (bool) wheather reaction is in graph"""
        return self.reaction_collection.has(reaction)

    def to_dict(self) -> dict:
        """Method to return the init parameters of the object as a dictionary"""
        return {
            "ac_matrix_type": self.ac_matrix_type,
            "specie_collection": type(self.specie_collection),
            "reaction_collection": type(self.reaction_collection),
            "specie_collection_kwargs": self.specie_collection.to_dict(),
            "reaction_collection_kwargs": self.reaction_collection.to_dict()
        }

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
            s.identifier = specie.identifier
            return s
        else:
            return specie

    def set_source_species(self, species: Iterable[Specie], force=False) -> None:
        """Method to set source species for the reaction graph.
        ARGS:
            - species (Iterable[Specie]): list of species to add to graph
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

    def to_networkx_graph(self, use_internal_id=False) -> nx.DiGraph:
        """Convert RxnGraph object to a bipartite networkx graph. Defaults for using the provided specie identifier and reaction index as node names.
        ARGS:
            - use_internal_id (bool): weather to use the internal id for reactions and species, for more robust conversion. default=False
        RETURNS:
            (nx.DiGraph) networkx bipartite graph representing the reaction graph"""
        # defining specie ID system
        if use_internal_id:
            get_id = lambda s: self.specie_collection.get_key(s)
        else:
            get_id = lambda s: s.identifier
        # initializing network
        G = nx.DiGraph()
        # adding species
        for i, s in enumerate(self.species):
            # adding the actual object to the networks graph
            G.add_node(get_id(s), obj=s)
        # adding reactions
        for i, rxn in enumerate(self.reactions):
            # adding reaction node
            if use_internal_id:
                rid = self.reaction_collection.get_key(rxn)
            else:
                rid = i
            G.add_node(rid, obj=rxn)
            # adding edges
            for s in rxn.reactants:
                G.add_edge(get_id(s), rid)
            for s in rxn.products:
                G.add_edge(rid, get_id(s))
        return G

    def _dfs_from_sources(self, network: nx.DiGraph, sources: Iterable[Specie]) -> Iterable[Reaction]:
        """Internal ethod to run a deep first search run from the network source to get all reactions connected to it.
        ARGS:
            - sources (Iterable[str]): list of graph sources
        RETURNS:
            (Generator[Reaction]) generator of reactions directly connected to the sources. the generator can yeild the same reaction few times"""
        for source in sources:
            if self.specie_collection.get_key(source) in network:
                for node in nx.algorithms.dfs_tree(network, self.specie_collection.get_key(source)).nodes:
                    if type(network.nodes[node]["obj"]) is Reaction:
                        yield network.nodes[node]["obj"]
    @TimeFunc
    def new(self, reactions: Optional[Iterable[Reaction]]=None, species: Optional[Iterable[Specie]]=None):
        """Method to make a new reaction graph with same definitions as the current one. if no reactions / species are specified, returns a copy
        ARGS:
            - reactions (List[Reaction]): list of reactions to add to new graph. default = None
            - species (List[Specie]): list of species to add to new graph. default = None
        RETURNS:
            (RxnGraph) a new reaction graph"""
        # parsing input arguments
        if reactions is None and species is None:
            reactions = self.reactions
            species = self.species
        elif reactions is None:
            reactions = []
        elif species is None:
            species = []
        # making an identical empty graph
        nrxn_graph = self.__class__(**self.to_dict())
        # copying reaction graph
        for specie in species:
            nrxn_graph.add_specie(specie)
        for rxn in reactions:
            nrxn_graph.add_reaction(rxn)
        # adding reactant species
        if not self.source_species is None:
            nrxn_graph.set_source_species(self.source_species, force=True)
        return nrxn_graph

    def _remove_objects(self, objects: Union[List[Specie], List[Reaction]], remove_neighbors: bool=False):
        """Internal method to remove an object (reaction or specie) from the reaction graph using deep first search"""
        if self.source_species is None:
            raise NotImplementedError("The graph doesnt have defined source reactants. This method is undefined in this case")
        # getting desired node
        if type(objects[0]) is Specie:
            nodes = [self.specie_collection.get_key(obj) for obj in objects]
        elif type(objects[0]) is Reaction:
            nodes = [self.reaction_collection.get_key(obj) for obj in objects]
        else:
            raise TypeError("Unsuported type for deletion {}".format(type(objects[0])))
        # convert to networkx graph
        network = self.to_networkx_graph(use_internal_id=True)
        # finding nodes to remove
        nodes_to_remove = []
        for node in nodes:
            nodes_to_remove.append(node)
            if remove_neighbors:
                nodes_to_remove += nx.all_neighbors(network, node)
        # removing nodes
        network.remove_nodes_from(nodes_to_remove)
        # running dfs to find other nodes to remove
        reactions = self._dfs_from_sources(network, self.source_species)
        return self.new(reactions=reactions)

    def remove_specie(self, specie: Specie):
        """Method to remove a specie from the reaction graph.
        ARGS:
            - specie (Specie): desired specie to remove
        RETURNS:
            (RxnGraph) a copy of the reaction graph without the speice"""
        # putting specie in format
        s = self._read_specie_with_ac_matrix(specie)
        # check if specie is in graph
        if not self.has_specie(s):
            raise ValueError("Desired specie is not in the graph !")
        # removing specie & all of its reactions (neighbors)
        return self._remove_objects([s], remove_neighbors=True)
    
    def remove_species(self, species: List[Specie]):
        """Method to remove a specie from the reaction graph.
        ARGS:
            - specie (Specie): desired specie to remove
        RETURNS:
            (RxnGraph) a copy of the reaction graph without the speice"""
        if len(species) == 0:
            return self
        _species = []
        for sp in species:
            # putting specie in format
            s = self._read_specie_with_ac_matrix(sp)
            # check if specie is in graph
            if not self.has_specie(s):
                raise ValueError("Desired specie is not in the graph !")
            _species.append(s)
        # removing specie & all of its reactions (neighbors)
        return self._remove_objects(_species, remove_neighbors=True)
    

    def remove_reaction(self, reaction: Reaction):
        """Method to remove a reaction from the reaction graph.
        ARGS:
            - reaction (Reaction): desired reaction to remove
        RETURNS:
            (RxnGraph) a copy of the reaction graph without the speice"""
        # check if specie is in graph
        if not self.has_reaction(reaction):
            raise ValueError("Desired reaction is not in the graph !")
        # removing reaction from graph
        return self._remove_objects([reaction], remove_neighbors=False)
    
    def remove_reactions(self, reactions: List[Reaction]):
        """Method to remove a reaction from the reaction graph.
        ARGS:
            - reaction (Reaction): desired reaction to remove
        RETURNS:
            (RxnGraph) a copy of the reaction graph without the speice"""
        if len(reactions) == 0:
            return self
        for r in reactions:
            # check if specie is in graph
            if not self.has_reaction(r):
                raise ValueError("Reaction {} is not in the graph !".format(r.pretty_string()))
        # removing reaction from graph
        return self._remove_objects(reactions, remove_neighbors=False)

    def join(self, rxn_graph) -> None:
        """Method to join a reaction graph into the current reaction graph.
        ARGS:
            - rxn_graph (RxnGraph): reaction graph to join with (adds its reactions to current graph)"""
        for rxn in rxn_graph.reactions:
            self.add_reaction(rxn)

    def _make_specie_df(self) -> pd.DataFrame:
        """Method to convert all specie data into a pd.DataFrame object. to be used for saving / reading reaction graph files"""
        res = pd.DataFrame()
        for i, s in enumerate(self.species):
            d = dict()
            d["idx"] = i
            d["ac_mat_str"] = s.ac_matrix._to_str()
            d["sid"] = self.specie_collection.get_key(s)
            d["identifier"] = s.identifier
            d["charge"] = s.charge
            d.update(s.properties)
            res = res.append(d, ignore_index=True)
        res = res.set_index("sid")
        return res

    def _make_reactions_df(self, species_df) -> pd.DataFrame:
        """Method to convert all reaction information into pd.DataFrame object. to be used for saving / reading reaction graph files"""
        res = pd.DataFrame()
        find_sp_idx = lambda sp: str(int(species_df.loc[self.specie_collection.get_key(sp), "idx"]))
        for r in self.reactions:
            d = {}
            d.update(r.properties)
            st = ",".join([find_sp_idx(sp) for sp in r.reactants]) + \
                    "=" + \
                    ",".join([find_sp_idx(sp) for sp in r.products])
            d["r_str"] = st
            res = res.append(d, ignore_index=True)
        if len(list(self.reactions)) > 0:
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
        with open(path_basename + "_params", "wb") as f:
            d = {"params": self.to_dict()}
            if self.source_species is not None:
                d["source_species_idxs"] = [int(specie_df.loc[self.specie_collection.get_key(sp), "idx"]) for sp in self.source_species]
            else:
                d["source_species_idxs"] = []
            d["class"] = self.__class__
            pickle.dump(d, f)
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
                d = pickle.load(f)
                rxn_graph = d["class"](**d["params"])
            # reading specie csv from zip to pd.DataFrame
            species_df = pd.read_csv(zipfile.open(path_basename + "_species"))
            source_sps = []
            for i, specie_row in enumerate(species_df.to_dict(orient="records")):
                # reading specie from ac_matrix string
                specie_mat = rxn_graph.ac_matrix_type()
                specie_mat._from_str(specie_row["ac_mat_str"])
                specie = specie_mat.to_specie()
                # reading charge
                specie.charge = None if pd.isna(specie_row["charge"]) else specie_row["charge"]
                # adding properties
                for c in specie_row.keys():
                    if not c in ["idx", "ac_mat_str", "charge", "sid"] and not pd.isna(specie_row[c]):
                        specie.properties[c] = specie_row[c]
                # adding specie to graph
                rxn_graph.add_specie(specie)
            # collecting all species
            species = list(rxn_graph.species)
            # setting source species
            rxn_graph.set_source_species([species[i] for i in d["source_species_idxs"]], force=True)
            # reading reactions csv from zip to pd.DataFrame
            reactions_df = pd.read_csv(zipfile.open(path_basename + "_reactions"))
            for rxn_row in reactions_df.to_dict(orient="records"):
                # reading reaction string
                r_str = rxn_row["r_str"]
                reactant_idxs = [int(s) for s in r_str.split("=")[0].split(",") if not len(s) == 0]
                reactants = [species[i] for i in reactant_idxs]
                product_idxs = [int(s) for s in r_str.split("=")[-1].split(",") if not len(s) == 0]
                products = [species[i] for i in product_idxs]
                rxn = Reaction(reactants, products)
                # adding properties
                for c in rxn_row.keys():
                    if not c in ["r_str"]:
                        rxn.properties[c] = rxn_row[c]
                # pushing to graph
                rxn_graph.add_reaction(rxn)
        return rxn_graph


    def make_specie_visited(self, specie: Specie):
        self.specie_collection.get(specie).properties["visited"] = True


    def get_unvisited_species(self) -> Iterable[Specie]:
        """Method to get all unvisited species in reaction graph"""
        for s in self.species:
            if not s.properties["visited"]:
                yield s

    def get_visited_species(self) -> Iterable[Specie]:
        """Method to get all unvisited species in reaction graph"""
        for s in self.species:
            if s.properties["visited"]:
                yield s

    def get_n_species(self) -> int:
        """Method to get the number of species in graph"""
        return len(self.specie_collection)

    def get_n_reactions(self) -> int:
            """Method to get the number of reactions in graph"""
            return len(self.reaction_collection)

    def __eq__(self, x):
        if not isinstance(x, RxnGraph):
            raise ValueError("Can compare only 2 reaction graphs")
        return len(list(self.reaction_collection.substract(x.reaction_collection))) == 0 and \
                len(list(x.reaction_collection.substract(self.reaction_collection))) == 0
