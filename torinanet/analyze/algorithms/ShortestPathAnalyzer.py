from typing import Callable, List, Optional
import pandas as pd
from .shortest_path_finders import dijkstra_shortest_path
from ...core.RxnGraph.RxnGraph import RxnGraph
from ...core.Specie import Specie


class ShortestPathAnalyzer:

    """Analyzer to get shortest path from every specie to graph source"""

    def __init__(self, rxn_graph: RxnGraph, shortest_path_finder=dijkstra_shortest_path, prop_func: Optional[Callable]=None) -> None:
        self.rxn_graph = rxn_graph
        self.networkx_graph = rxn_graph.to_networkx_graph(use_internal_id=True)
        self.shortest_path_table = shortest_path_finder(rxn_graph, prop_func)


    def get_distance_from_source(self, specie: Specie):
        if self.rxn_graph.has_specie(specie):
            s_id = self.rxn_graph.specie_collection.get_key(specie)
            return self.shortest_path_table.loc[s_id, "dist"]
        else:
            raise ValueError("requested specie is not in graph")

    
    def get_path_to_source(self, specie: Specie):
        target = self.rxn_graph.specie_collection.get_key(specie)
        rxn_path = [self.shortest_path_table.loc[target, "rxn"]]
        # in case source specie is queried, return empty path
        if rxn_path[0] is None:
            return []
        while True:
            parent_rxn = rxn_path[-1]
            # finding the predecessor reactant (one with max distance)
            max_dist = None
            parent_specie = None
            for reactant in self.networkx_graph.predecessors(self.rxn_graph.reaction_collection.get_key(parent_rxn)):
                dist = self.shortest_path_table.loc[reactant, "dist"]
                if max_dist is None:
                    max_dist = dist
                    parent_specie = reactant
                if dist > max_dist:
                    parent_specie = reactant
                    max_dist = dist
            new_rxn = self.shortest_path_table.loc[parent_specie, "rxn"]
            # stopping condition - if new reaction exists in path
            if new_rxn in rxn_path or new_rxn is None:
                break
            # adding the predecessor's reaction to the rxn_path
            rxn_path.append(new_rxn)
        return rxn_path


    def get_species_with_dist(self, distance: int) -> List[Specie]:
        """Method to get a list of species with a desired distance from the source species.
        ARGS:
            - distance (int): desired distance
        RETURNS:
            (List[Specie]) list of species with distance. returns empty list if no specie is found"""
        # list of species IDs that are within the distance
        return self.shortest_path_table[self.shortest_path_table["dist"] == distance]["specie"]


    def get_distance_distribution(self, species: Optional[List[Specie]]=None) -> dict:
        """Method to get a distance distribution of species.
        ARGS:
            - species (Optional[List[Specie]]): list of species to make distribution for. default=None (all species)
        RETURNS:
            (dict) dictionary with distance (keys) and number of species (values)"""
        # reading specie ids
        if species:
            sids = [self.rxn_graph.specie_collection.get_key(s) for s in species]
        else:
            sids = self.shortest_path_table.index
        # grouping distance table
        res = self.shortest_path_table.loc[sids, :].groupby(["dist"]).count()
        # returning results as a dictionary
        return res.to_dict().values()[0]
        
    def to_dataframe(self) -> pd.DataFrame:
        """Method to convert the analysis results into a human-readable dataframe"""
        df = pd.DataFrame()
        df["smiles"] = [s.identifier for s in self.shortest_path_table["specie"]]
        df["dist"] = [self.get_distance_from_source(s) for s in self.shortest_path_table["specie"]]
        df["n_reactions"] = [len(self.get_path_to_source(s)) for s in self.shortest_path_table["specie"]]
        return df

    def number_of_paths_per_specie(self) -> pd.DataFrame():
        """Get the number of paths passing through each specie in the graph"""
        # counter dictionary with data on all species in graphs, we distinguish between "created" paths - where the specie is created in the path (in the products) or "consumed" paths - where specie is consumed in the path (in the reactants) or, "intermediate" - when it is not in the total reaction but appears in the path steps
        res = {sp.identifier: {"consumed": 0, "created": 0, "intermediate": 0} for sp in self.rxn_graph.species}
        for sp in self.rxn_graph.species:
            # finding the shortest path
            path = self.get_path_to_source(sp)
            # counting the "total reaction" from source to target for proper accounting of "created" and "consumed" species
            ajr = {}
            # getting the "total reaction" getting the coefficients of all species in the path - if negative - specie is reactant, else it is product
            for rxn in path:
                for s in rxn.reactants:
                    smiles = s.identifier
                    if not smiles in ajr:
                        ajr[smiles] = -1
                    else:
                        ajr[smiles] += -1
                for s in rxn.products:
                    smiles = s.identifier
                    if not smiles in ajr:
                        ajr[smiles] = 1
                    else:
                        ajr[smiles] += 1
            for smiles, coeff in ajr.items():
                if coeff < 0:
                    res[smiles]["consumed"] += 1
                elif coeff == 0:
                    res[smiles]["intermediate"] += 1
                else:
                    res[smiles]["created"] += 1
        return pd.DataFrame({"smiles": res.keys(), "consumed": [d["consumed"] for d in res.values()], "created": [d["created"] for d in res.values()], "intermediate": [d["intermediate"] for d in res.values()]})

