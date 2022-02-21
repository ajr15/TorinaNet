import pandas as pd
from time import time
import sys; sys.path.append("../..")

from src.iterate.filters.ac_matrix_filters import MaxBondsPerAtom
from src.iterate.filters.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds
from src.iterate.stop_conditions import MaxIterNumber
from src.core import Specie


def iterate_module_standard_test(smiles, rxn_graph_module, iterator_module):
    rxn_graph = rxn_graph_module()
    rxn_graph.add_specie(Specie(smiles))
    stop_cond = MaxIterNumber(2)
    ac_filters = [MaxBondsPerAtom()]
    conversion_filters = [MaxChangingBonds(3), OnlySingleBonds()]
    iterator = iterator_module(rxn_graph)
    rxn_graph = iterator.enumerate_reactions(conversion_filters, ac_filters, stop_cond, verbose=1)


def main():
    import src.iterate
    src.iterate.kernel = "vanilla"
    from src.iterate.iterators import Iterator
    from src.core.RxnGraph import daskRxnGraph
    res = pd.DataFrame()
    for smiles in ["C", "CC"]:
        print(smiles)
        t1 = time()
        iterate_module_standard_test(smiles, daskRxnGraph, Iterator)
        t2 = time()
        res = res.append({"smiles": smiles, "time": t2 - t1}, ignore_index=True)
    print(res)



if __name__ == "__main__":
    main()