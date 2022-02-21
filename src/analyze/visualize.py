from networkx.algorithms import sparsifiers
from ..core.RxnGraph.RxnGraph import RxnGraph
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np
from typing import Optional

def choose(n: int, target: list):
    idxs = set()
    res = []
    while True:
        i = np.random.randint(0, len(target) - 1)
        if not i in idxs:
            idxs.add(i)
            res.append(target[i])
        if len(res) == n:
            return res



def apply_func_on_species(rxn_graph: RxnGraph, func: callable, njobs: int, sample_n: Optional[int]=None):
    if sample_n is None:
        species = rxn_graph.species
    else:
        species = choose(sample_n, rxn_graph.species)
    if njobs > 1:
        with multiprocessing.Pool(njobs) as pool:
            return pool.map(func, species)
    else:
        return [func(s) for s in species]


def specie_func_distribution(rxn_graph: RxnGraph, prop_func: callable, sample_n: Optional[int]=None, title: str="", x_label: str="", njobs=1, **hist_kwargs):
    vals = apply_func_on_species(rxn_graph, prop_func, njobs, sample_n)
    plt.figure()
    plt.hist(vals, **hist_kwargs)
    plt.title(title)
    plt.xlabel(x_label)