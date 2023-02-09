from ...core.RxnGraph import RxnGraph
from typing import Callable, Optional
import numpy as np
import pandas as pd
import networkx as nx

def dijkstra_shortest_path(rxn_graph: RxnGraph, prop_func: Optional[Callable]=None):
    """Method to implement modified dijkstra's algorithm for finding distance between specie to the source of the network (reactants)"""
    # making property function
    if not prop_func:
        prop_func = lambda rxn: 1
    # converting reaction graph to networkx
    G = rxn_graph.to_networkx_graph(use_internal_id=True)
    graph_objects = nx.get_node_attributes(G, "obj")
    source_species = [rxn_graph.specie_collection.get_key(s) for s in rxn_graph.source_species]
    # initializing 
    # dictionary for each specie its distance from source and making reaction
    specie_df = {rxn_graph.specie_collection.get_key(s):
                     {"dist": np.inf, "rxn": None, "visited": False, "specie": s} for s in rxn_graph.species}
    for s in source_species:
        specie_df[s].update({"dist": 0, "rxn": None, "visited": False})
    specie_df = pd.DataFrame.from_dict(specie_df, orient="index")
    # running main loop
    while not all(specie_df["visited"].values): # runs until all species are visited
        # find the next specie to visit (unvisited specie with shortest distance from source)
        unvisited = specie_df[specie_df["visited"] == False]
        specie = unvisited[unvisited["dist"] == unvisited["dist"].min()].first_valid_index()
        # chaning flag to visited
        specie_df.loc[specie, "visited"] = True
        # go over all reactions with visited reactants
        for rxn in G.successors(specie):
            pred_species = [s for s in G.predecessors(rxn)]
            if all([specie_df.loc[s, "visited"] for s in pred_species]):
                # make a distance estimate for the reaction's products
                # the estimate is the maximal distance of the reactant species plus one
                prod_species = [s for s in G.successors(rxn)]
                reaction = graph_objects[rxn]
                dist_estimate = max([specie_df.loc[s, "dist"] for s in pred_species]) + prop_func(reaction)
                for s in prod_species:
                    # if the distance estimate is less than the known distance, update distance and pred reaction
                    if dist_estimate < specie_df.loc[s, "dist"]:
                        specie_df.loc[s, "dist"] = dist_estimate
                        specie_df.loc[s, "rxn"] = reaction
    # dropping the "visited column"
    specie_df = specie_df.drop(columns=["visited"])
    return specie_df
