from time import time
from numba import njit
import numpy as np
import sys; sys.path.append("../..")
from torinanet.iterate.filters.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds
import torinanet.iterate.utils as utils
from torinanet.iterate.filters.ac_matrix_filters import max_bonds_per_atom

def compare_jitted_version(f, jitted, *args, **kwargs):
    print("====== WITHOUT JIT ======")
    avg_t_no_jit = 0
    for i in range(3):
        tick = time()
        f(*args, **kwargs)
        tock = time()
        print(tock - tick)
        avg_t_no_jit += (tock - tick)
    avg_t_no_jit = avg_t_no_jit / 3
    print("====== WITH JIT ======")
    print("compiling...")
    jitted(*args, **kwargs)
    print("running test")
    avg_t_jit = 0
    for i in range(3):
        tick = time()
        jitted(*args, **kwargs)
        tock = time()
        print(tock - tick)
        avg_t_jit += (tock - tick)
    avg_t_jit = avg_t_jit / 3
    print("average without JIT:", round(avg_t_no_jit, 4))
    print("average with JIT:", round(avg_t_jit, 4))
    print("RATIO:", round(avg_t_no_jit / avg_t_jit) if not avg_t_jit == 0 else "UNDEFINED")

def apply_single_filter_test():
    test_mat = np.random.randint(0, 1, size=(10000, 7, 7), dtype=np.int8)
    bool_range = np.ones(test_mat.shape[0])
    filter_func = njit() (lambda m: np.sum(m) == 7)
    jitted = njit() (utils.apply_single_filter)
    compare_jitted_version(utils.apply_single_filter, jitted, bool_range, test_mat, filter_func)

def generate_single_conversion_matrices_without_filter_test():
    test_ac = np.random.randint(0, 1, size=(200, 200), dtype=np.int8)
    jitted = njit() (utils.generate_single_conversion_matrices_without_filter)
    compare_jitted_version(utils.generate_single_conversion_matrices_without_filter,
                            jitted, 
                            test_ac,
                            True)

def filter_by_boolian_test():
    test_ac = np.random.randint(0, 2, size=(2000000, 7, 7), dtype=np.int8)
    boolian = np.random.randint(0, 2, 2000000, dtype=bool)
    jitted = njit() (utils.filter_by_boolian)
    compare_jitted_version(utils.filter_by_boolian, 
                            jitted,
                            boolian,
                            test_ac)


def generate_conversion_matrices_test():
    ac = np.random.choice([1, 0], (13, 13), p=[0.1, 0.9])
    conversion_filters = [OnlySingleBonds(), MaxChangingBonds(4)]
    
    def test_func(ac, filters):
        for x in utils.generate_conversion_matrices(ac, filters):
            x

    def jit_test_func(ac, filters):
        for x in utils.jit_generate_conversion_matrices(ac, filters):
            x

    compare_jitted_version(test_func, 
                            jit_test_func,
                            ac, 
                            conversion_filters)


def iterate_over_ac_test():
    from torinanet.core.Specie import Specie
    from torinanet.core.AcMatrix.BinaryAcMatrix import BinaryAcMatrix
    ac = BinaryAcMatrix.from_specie(Specie("CCCO")).matrix
    conversion_filters = [OnlySingleBonds(), MaxChangingBonds(4)]
    ac_filters = [utils.max_bonds_per_atom]
    
    def test_func(ac):
        for res in utils.iterate_over_ac_matrix(ac, conversion_filters, ac_filters):
            res
    
    def jit_test_func(ac):
        for res in utils.jit_iterate_over_ac_matrix(ac, conversion_filters, ac_filters):
            res
        
    compare_jitted_version(test_func, jit_test_func, ac)
    

if __name__ == "__main__":
    iterate_over_ac_test()
