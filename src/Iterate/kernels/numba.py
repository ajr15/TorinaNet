import numpy as np
from numba import njit

from ..filters.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds, _TwoSpecieMatrix
from .commons import gen_unique_idx_combinations_for_conv_mats


@njit
def generate_single_conversion_matrices_without_filter(ac, only_single_bonds):
    """Internal method to generate single conversion matrices using numba accelarator"""
    ac_l = len(ac)
    if len(ac) < 2:
        conv_matrices = np.zeros((0, 0, 0), dtype=np.int8)
    conv_matrices = np.empty((int((ac_l * (ac_l - 1)) / 2), ac_l, ac_l), dtype=np.int8)
    for i in range(len(ac)):
        for j in range(len(ac) - i - 1):
            if ac[i][j + i + 1] == 0: # makes sure there are no negative bond orders
                mat = np.zeros((len(ac), len(ac)))
                mat[i][j + i + 1] = 1
                mat[j + i + 1][i] = 1
                conv_matrices[int((ac_l - 1 - (i - 1) / 2) * i + j)] = mat
            else: # makes sure only single bonds are created
                mat = np.zeros((len(ac), len(ac)))
                mat[i][j + i + 1] = -1
                mat[j + i + 1][i] = -1
                conv_matrices[int((ac_l - 1 - (i - 1) / 2) * i + j)] = mat
                if not only_single_bonds:
                    mat = np.zeros((len(ac), len(ac)))
                    mat[i][j + i + 1] = 1
                    mat[j + i + 1][i] = 1
                    conv_matrices[int((ac_l - 1 - (i - 1) / 2) * i + j)] = mat
    return conv_matrices


def conversion_matrix_generator(single_conv_mats, idxs_for_comb):
    for idxs in idxs_for_comb:
        yield sum_by_idxs(single_conv_mats, np.array(idxs, dtype=np.int64))


@njit
def sum_by_idxs(single_conv_mats, idxs):
    mat = single_conv_mats[idxs[0]].copy()
    for idx in idxs[1:]:
        if not idx == -1:
            mat += single_conv_mats[idx]
    return mat


@njit
def only_single_bonds(matrix):
    for v in matrix:
        for x in v:
            if not (not x > 1 and not x < -1):
                return False
    return True


def jit_conversion_filters(filters):
    jitted = []
    for f in filters:
        if isinstance(f, OnlySingleBonds):
            jitted.append(only_single_bonds)
        elif isinstance(f, _TwoSpecieMatrix):
            l = f.max_l
            fi = lambda mat: np.sum(mat[:l, l:]) > 0
            jitted.append(njit() (fi))
        elif isinstance(f, MaxChangingBonds):
            continue
        else:
            raise NotImplementedError("Not all of the specified conversion matrix filters are supported with Numbe-JIT, \
                                        please switch to the vanila version or remove filter")
    return jitted


def generate_conversion_matrices(ac, conversion_filters):
    if not any([isinstance(x, MaxChangingBonds) for x in conversion_filters]):
        raise ValueError("Must set a maximum number of changing bonds in conversion filters")
    # get information on filters
    only_single_bonds = False
    for x in conversion_filters:
        # maximum changing bonds
        if isinstance(x, MaxChangingBonds):
            max_changing_bonds = x.n
        # allow only single bonds?
        if isinstance(x, OnlySingleBonds):
            only_single_bonds = True
    conversion_filters = jit_conversion_filters(conversion_filters)
    conv_matrices = generate_single_conversion_matrices_without_filter(ac, only_single_bonds)
    conv_matrices = filter_matrices(conv_matrices, conversion_filters)
    n_conv_matrices = len(conv_matrices) # number of single bond transformations, used to add unique matrices in iteration step
    # iterate on conv_matrices (single bond changes) to get multi-bond change conversion matrices
    idx_combinations = gen_unique_idx_combinations_for_conv_mats(n_conv_matrices, max_changing_bonds)
    for mat in conversion_matrix_generator(conv_matrices, idx_combinations):
        if all([f(mat) for f in conversion_filters]):
            yield mat


@njit
def apply_single_filter(bool_range, matrices, f):
    out = np.empty_like(bool_range)
    for i in range(len(matrices)):
        if not bool_range[i]:
            out[i] = False
        else:
            out[i] = f(matrices[i])
    return out


def apply_filters_on_matrices(matrices, filters):
    bool_range = np.ones(len(matrices), dtype=bool)
    for f in filters:
        bool_range = apply_single_filter(bool_range, matrices, f)
    return bool_range


@njit
def filter_by_boolian(bool_range, target):
    filtered = np.empty((np.sum(bool_range), *target.shape[1:]), dtype=np.int8)
    idx = 0
    for b, t in zip(bool_range, target):
        if b:
            filtered[idx] = t
            idx += 1
    return filtered


def filter_matrices(matrices, filters):
    boolian = apply_filters_on_matrices(matrices, filters)
    return filter_by_boolian(boolian, matrices)


def enumerate_over_ac(ac, conv_filters, ac_filters):
    ac_type = ac.__class__
    conv_mats = generate_conversion_matrices(ac.matrix, conv_filters)
    for conv_mat in conv_mats:
        new_ac = ac_type(ac.matrix.copy() + conv_mat)
        if all([f.check(new_ac) for f in ac_filters]):
            yield new_ac