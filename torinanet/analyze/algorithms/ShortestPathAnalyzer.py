from typing import Callable, List, Optional

from ...core.Reaction import Reaction
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
        s_id = self.rxn_graph.make_unique_id(specie)
        return self.shortest_path_table.loc[s_id, "dist"]

    
    def get_path_to_source(self, specie: Specie):
        target = specie._get_id_str()
        rxn_path = [self.shortest_path_table.loc[target, "rxn"]]
        while True:
            parent_rxn = rxn_path[-1]
            # finding the predecessor reactant (one with max distance)
            max_dist = 0
            parent_specie = None
            for reactant in self.networkx_graph.predecessors(parent_rxn):
                dist = self.shortest_path_table.loc[reactant, "dist"]
                if dist > max_dist:
                    parent_specie = reactant
                    max_dist = dist
            # stoping condition = the predecessor is a source specie
            if max_dist == 0:
                break
            # adding the predecessor's reaction to the rxn_path
            new_rxn = self.shortest_path_table.loc[parent_specie, "rxn"]
            rxn_path.append(new_rxn)
        return [self.rxn_graph.get_reaction_from_id(rid) for rid in rxn_path]


    def get_species_with_dist(self, distance: int) -> List[Specie]:
        """Method to get a list of species with a desired distance from the source species.
        ARGS:
            - distance (int): desired distance
        RETURNS:
            (List[Specie]) list of species with distance. returns empty list if no specie is found"""
        # list of species IDs that are within the distance
        sids = self.shortest_path_table[self.shortest_path_table["dist"] == distance].index
        return [self.rxn_graph.get_specie_from_id(sid) for sid in sids]


    def get_distance_distribution(self, species: Optional[List[Specie]]=None) -> dict:
        """Method to get a distance distribution of species.
        ARGS:
            - species (Optional[List[Specie]]): list of species to make distribution for. default=None (all species)
        RETURNS:
            (dict) dictionary with distance (keys) and number of species (values)"""
        # reading specie ids
        if species:
            sids = [self.rxn_graph.make_unique_id(s) for s in species]
        else:
            sids = self.shortest_path_table.index
        # grouping distance table
        res = self.shortest_path_table.loc[sids, :].groupby(["dist"]).count()
        # returning results as a dictionary
        return res.to_dict().values()[0]
        