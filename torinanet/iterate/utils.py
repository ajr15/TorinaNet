import numpy as np
from numba import njit
from itertools import combinations
from .filters.conversion_matrix_filters import OnlySingleBonds, _TwoSpecieMatrix, MaxChangingBonds

@njit
def only_single_bonds(matrix):
    for v in matrix:
        for x in v:
            if not (not x > 1 and not x < -1):
                return False
    return True


@njit
def _jitted_ac_filter(ac_matrix):
    """Filter that makes sure that atoms don't have too many bonds in the ac matrix"""
    max_changing_bonds_dict = {
                                1: 1,
                                6: 4,
                                7: 4,
                                8: 2,
                                9: 1
                                }

    for i in range(len(ac_matrix)):
        max_changing_bonds_dict[ac_matrix[i][i]]
        if max_changing_bonds_dict[ac_matrix[i][i]] < np.sum(ac_matrix[i]) - ac_matrix[i][i]:
            return False
    return True


def max_bonds_per_atom(ac_matrix):
    """Filter that makes sure that atoms don't have too many bonds in the ac matrix"""
    max_changing_bonds_dict = {
                                1: 1,
                                6: 4,
                                7: 4,
                                8: 2,
                                9: 1
                                }

    for i in range(len(ac_matrix)):
        if max_changing_bonds_dict[ac_matrix[i][i]] < np.sum(ac_matrix[i]) - ac_matrix[i][i]:
            return False
    return True


def jit_ac_filters(filters):
    jitted = []
    for f in filters:
        if f.__name__ == "max_bonds_per_atom":
            jitted.append(_jitted_ac_filter)
        else:
            raise NotImplementedError("ac matrix filter is not supported in Numba-JIT, please use vanilla version or remove filter")
    return jitted


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


def gen_unique_idx_combinations_for_conv_mats(n_single_conv_mats, max_changing_bonds):
    for n in range(1, max_changing_bonds + 1):
        for c in combinations(range(n_single_conv_mats), n):
            yield list(c)


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

jit_generate_single_conversion_matrices_without_filter = njit() (generate_single_conversion_matrices_without_filter)


def conversion_matrix_generator(single_conv_mats, idxs_for_comb):
    """Generator for conversion matrices given single conversion matrices and indicis for combination. filters results using single filter. returns result conv matrix or None"""
    conv_shape = single_conv_mats.shape[1:]
    for idxs in idxs_for_comb:
        mat = np.zeros(conv_shape)
        for idx in idxs:
            mat += single_conv_mats[idx]
        yield mat


def jit_conversion_matrix_generator(single_conv_mats, idxs_for_comb):
    for idxs in idxs_for_comb:
        yield jit_sum_by_idxs(single_conv_mats, np.array(idxs, dtype=np.int64))

@njit
def jit_sum_by_idxs(single_conv_mats, idxs):
    mat = single_conv_mats[idxs[0]]
    for idx in idxs[1:]:
        if not idx == -1:
            mat += single_conv_mats[idx]
    return mat



def iterate_over_ac_matrix(ac, conv_filters, ac_filters):
    """Method to iterate over single AC matrix using conversion matrix generator. filters using single filter, returns result ac matrix or None"""
    conv_mats_generator = generate_conversion_matrices(ac, conv_filters)
    for conv_mat in conv_mats_generator:
        mat = ac + conv_mat
        if all([f(mat) for f in ac_filters]):
            yield mat


def jit_iterate_over_ac_matrix(ac, conv_filters, ac_filters):
    conv_mats_generator = jit_generate_conversion_matrices(ac, conv_filters)
    for conv_mat in conv_mats_generator:
        mat = ac + conv_mat
        if all([f(mat) for f in ac_filters]):
            yield mat

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
    conv_matrices = generate_single_conversion_matrices_without_filter(ac, only_single_bonds)
    conv_matrices = filter_matrices(conv_matrices, conversion_filters)
    n_conv_matrices = len(conv_matrices) # number of single bond transformations, used to add unique matrices in iteration step
    # iterate on conv_matrices (single bond changes) to get multi-bond change conversion matrices
    idx_combinations = gen_unique_idx_combinations_for_conv_mats(n_conv_matrices, max_changing_bonds)
    for mat in conversion_matrix_generator(conv_matrices, idx_combinations):
        if all([f.check(mat) for f in conversion_filters]):
            yield mat

def jit_generate_conversion_matrices(ac, conversion_filters):
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
    conv_matrices = jit_generate_single_conversion_matrices_without_filter(ac, only_single_bonds)
    conv_matrices = jit_filter_matrices(conv_matrices, conversion_filters)
    n_conv_matrices = len(conv_matrices) # number of single bond transformations, used to add unique matrices in iteration step
    # iterate on conv_matrices (single bond changes) to get multi-bond change conversion matrices
    idx_combinations = gen_unique_idx_combinations_for_conv_mats(n_conv_matrices, max_changing_bonds)
    for mat in jit_conversion_matrix_generator(conv_matrices, idx_combinations):
        if all([f(mat) for f in conversion_filters]):
            yield mat


# ==================================================
#           POSIBLY USELESS FUNCTIONS
# ==================================================

def apply_single_filter(bool_range, matrices, f):
    out = np.empty_like(bool_range)
    for i in range(len(matrices)):
        if not bool_range[i]:
            out[i] = False
        else:
            out[i] = f(matrices[i])
    return out

jit_apply_single_filter = njit() (apply_single_filter)


def apply_filters_on_matrices(matrices, filters):
    bool_range = np.ones(len(matrices), dtype=bool)
    for f in filters:
        bool_range = apply_single_filter(bool_range, matrices, f.check)
    return bool_range

def jit_apply_filters_on_matrices(matrices, filters):
    bool_range = np.ones(len(matrices), dtype=bool)
    for f in filters:
        bool_range = jit_apply_single_filter(bool_range, matrices, f)
    return bool_range


def filter_by_boolian(bool_range, target):
    filtered = np.empty((np.sum(bool_range), *target.shape[1:]), dtype=np.int8)
    idx = 0
    for b, t in zip(bool_range, target):
        if b:
            filtered[idx] = t
            idx += 1
    return filtered

jit_filter_by_boolian = njit() (filter_by_boolian)

def filter_matrices(matrices, filters):
    boolian = apply_filters_on_matrices(matrices, filters)
    return filter_by_boolian(boolian, matrices)

def jit_filter_matrices(matrices, filters):
    boolian = jit_apply_filters_on_matrices (matrices, filters)
    return jit_filter_by_boolian(boolian, matrices)
