import sys; sys.path.append("../")
import torinanet as tn
from dask.distributed import Client
from multiprocessing import Pool
from utils import specie_generator
from time import time
from itertools import combinations
import pandas as pd

def run(source_species, scheduler):
    g = tn.core.RxnGraph()
    g.set_source_species(source_species, force=True)
    iterator = tn.iterate.Iterator(g, dask_scheduler=scheduler)
    conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                            tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
    ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                    tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
    stop_condition = tn.iterate.stop_conditions.MaxIterNumber(1)
    iterator.enumerate_reactions(conv_filters, ac_filters, stopping_condition=stop_condition)

def wrapper(args):
    print("Running", args[1])
    args[0].iterate_over_species(*args[2])

def test_multiprocessing(source_species, n_workers):
    print("RUNNING WITH {} WORKERS".format(n_workers))
    iterator = tn.iterate.Iterator(tn.core.RxnGraph())
    conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                            tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
    ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                    tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
    args = []
    for i, (s1, s2) in enumerate(combinations(source_species, 2)):
        args.append((iterator, i, (s1, s2, ac_filters, conv_filters)))
    print(len(args))
    ti = time()
    Pool(n_workers).map(wrapper, args)
    return time() - ti

    
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scheduler", type=str)
    args = parser.parse_args()
    smiles = ["CCC"]
    source_species = []
    print("SOURCE SPECIES")
    for sp in specie_generator([tn.core.Specie(smiles_str) for smiles_str in smiles], 2, 0):
        source_species.append(sp)
        print(sp.identifier)
        if len(source_species) > 13:
            break
    res = pd.DataFrame()
    t1 = time()
    if args.scheduler == "distributed":
        with Client():
            run(source_species, args.scheduler)
    else:
        run(source_species, args.scheduler)
    print("TOTAL TIME =", time() - t1)


if __name__ == "__main__":
    smiles = ["CC"]
    source_species = [tn.core.Specie("C")] * 5
    # print("SOURCE SPECIES")
    # for sp in specie_generator([tn.core.Specie(smiles_str) for smiles_str in smiles], 2, 0):
    #     source_species.append(sp)
    #     print(sp.identifier)
    #     if len(source_species) > 5:
    #         break
    res = pd.DataFrame()
    for n in [2, 4, 8]:
        d = {"N": n}
        d["time"] = test_multiprocessing(source_species, n)
        res = res.append(d, ignore_index=True)
        res.to_csv("timing.csv")