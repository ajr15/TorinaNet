import sys; sys.path.append("../")
import torinanet as tn
from utils import specie_generator
from time import time

def run(source_species, n_workers):
    g = tn.core.RxnGraph()
    g.set_source_species(source_species, force=True)
    iterator = tn.iterate.Iterator(g, n_workers)
    conv_filters = [tn.iterate.conversion_matrix_filters.MaxChangingBonds(3),
                            tn.iterate.conversion_matrix_filters.OnlySingleBonds()]
    ac_filters = [tn.iterate.ac_matrix_filters.MaxBondsPerAtom(), 
                    tn.iterate.ac_matrix_filters.MaxAtomsOfElement({6: 4, 8: 4, 7: 4})]
    stop_condition = tn.iterate.stop_conditions.MaxIterNumber(1)
    iterator.enumerate_reactions(conv_filters, ac_filters, stopping_condition=stop_condition)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n_workers", type=int)
    args = parser.parse_args()
    smiles = ["C"]
    source_species = []
    print("SOURCE SPECIES")
    for sp in specie_generator([tn.core.Specie(smiles_str) for smiles_str in smiles], 2, 0):
        source_species.append(sp)
        print(sp.identifier)
        if len(source_species) > 13:
            break
    t1 = time()
    run(source_species, args.n_workers)
    print("TOTAL TIME =", time() - t1)

if __name__ == "__main__":
    main()