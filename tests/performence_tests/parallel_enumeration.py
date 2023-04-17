import torinanet as tn
from dask.distributed import Client
from time import time
from itertools import product
import pandas as pd

def run(reactants_smiles, n_workers, scheduler):
    titles = ["n_workers", "scheduler"]
    combo = [n_workers, scheduler]
    print(", ".join(["{}={}".format(t, c) for t, c in zip(titles, combo)]))
    g = tn.core.RxnGraph()
    g.set_source_species([tn.core.Specie(s) for s in reactants_smiles], force=True)
    iterator = tn.iterate.Iterator(g, dask_scheduler=scheduler)
    conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                            tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
    ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                    tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
    stop_condition = tn.iterate.stop_conditions.MaxIterNumber(1)
    with Client(n_workers=n_workers):
        t1 = time()
        iterator.enumerate_reactions(conv_filters, ac_filters, stopping_condition=stop_condition)
        d = {t: c for t, c in zip(titles, combo)}
        d["time"] = time() - t1
        return d
    
if __name__ == "__main__":
    combos = [
        [1, "synchronous"],
        # [2, "threads", 200, 500, 2000],
        [4, "threads"],
        # [8, "threads", 200, 500, 2000],
        # [16, "threads", 200, 500, 2000],
    ]
    smiles = ["CC"]
    res = pd.DataFrame()
    for combo in combos:
        d = run(smiles, *combo)
        res = res.append(d, ignore_index=True)
        res.to_csv("results.csv")