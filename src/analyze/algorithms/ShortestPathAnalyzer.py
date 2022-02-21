from .shortest_path_finders import dijkstra_shortest_path
from ...core.RxnGraph.BaseRxnGraph import BaseRxnGraph
from ...core.Specie import Specie


class ShortestPathAnalyzer:

    """Analyzer to get shortest path from every specie to graph source"""

    def __init__(self, rxn_graph: BaseRxnGraph, shortest_path_finder=dijkstra_shortest_path) -> None:
        self.rxn_graph = rxn_graph
        self.networkx_graph = rxn_graph.to_networkx_graph(use_internal_id=True)
        self.shortest_path_table = shortest_path_finder(rxn_graph)


    def get_distance_from_source(self, specie: Specie):
        s_id = specie._get_id_str()
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
