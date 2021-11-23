import sys; sys.path.append("../..")
import numpy as np
from src.Iterate.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds, _TwoSpecieMatrix
from src.Iterate._jitted_commons import jit_conversion_filters, _apply_filters_on_matrices, _make_multibond_conv_matrices

def apply_filters_test():
    f = jit_conversion_filters([MaxChangingBonds(2), OnlySingleBonds()])
    data = np.random.randint(0, 2, size=(100, 10, 10))
    res = _apply_filters_on_matrices(data, f)

    f = jit_conversion_filters([MaxChangingBonds(2), OnlySingleBonds(), _TwoSpecieMatrix(6)])
    new_data = np.random.randint(0, 2, size=(500, 15, 15))
    res = _apply_filters_on_matrices(data, f)


def multi_conv_mats_test():
    conv_matrices = np.random.randint(0, 2, size=(100, 10, 10))
    idxs = np.random.randint(0, len(conv_matrices) - 1, size=(50, 4))
    multi_matrices = _make_multibond_conv_matrices(idxs, conv_matrices)
    print("did first")

    conv_matrices = np.random.randint(0, 2, size=(100, 15, 15))
    idxs = np.random.randint(0, len(conv_matrices) - 1, size=(50, 4))
    multi_matrices = _make_multibond_conv_matrices(idxs, conv_matrices)
    print("did second")

if __name__ == '__main__':
    apply_filters_test()