from abc import ABC, abstractclassmethod
import numpy as np
from numba import njit
from itertools import product
from .conversion_matrix_filters import OnlySingleBonds, _TwoSpecieMatrix, MaxChangingBonds

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

@njit
def _generate_single_conversion_matrices_without_filter(ac, only_single_bonds):
    """Internal method to generate single conversion matrices using numba accelarator"""
    ac_l = len(ac)
    if len(ac) < 2:
        return np.zeros((0, 0, 0), dtype=np.int8)
    conv_matrices = np.zeros((int((ac_l * (ac_l - 1)) / 2 + 1), ac_l, ac_l), dtype=np.int8)
    for i in range(len(ac)):
        for j in range(i + 1, len(ac)):
            if ac[i][j] == 0: # makes sure there are no negative bond orders
                mat = np.zeros((len(ac), len(ac)))
                mat[i][j] = 1
                mat[j][i] = 1
                conv_matrices[(i + 1) * j + 1] = mat
            else: # makes sure only single bonds are created
                mat = np.zeros((len(ac), len(ac)))
                mat[i][j] = -1
                mat[j][i] = -1
                conv_matrices[(i + 1) * j + 1] = mat
                if not only_single_bonds:
                    mat = np.zeros((len(ac), len(ac)))
                    mat[i][j] = 1
                    mat[j][i] = 1
                    conv_matrices[(i + 1) * j + 1] = mat
    return conv_matrices

@njit
def _apply_single_filter(bool_range, matrices, f):
    new_bool = np.ones(len(matrices), dtype=np.bool8)
    for i in range(len(matrices)):
        if not bool_range[i]:
            new_bool[i] = False
        else:
            new_bool[i] = f(matrices[i])
    return new_bool

def _apply_filters_on_matrices(matrices, filters):
    bool_range = np.ones(len(matrices), dtype=np.bool8)
    for f in filters:
        bool_range = _apply_single_filter(bool_range, matrices, f)
    return bool_range

@njit
def _filter_by_boolian(bool_range, target):
    filtered = np.zeros((np.sum(bool_range), *target.shape[1:]), dtype=np.int8)
    idx = 0
    for b, t in zip(bool_range, target):
        if b:
            filtered[idx] = t
            idx += 1
    return filtered

@njit()
def _make_multibond_conv_matrices(single_bond_mats_idxs, single_bond_mats):
    mats = np.zeros((len(single_bond_mats_idxs), *single_bond_mats.shape[1:]), dtype=np.int8)
    if len(single_bond_mats_idxs) == 0:
        return mats
    for i in range(len(single_bond_mats_idxs)):
        for idx in single_bond_mats_idxs[i]:
            mats[i] = mats[i] + single_bond_mats[idx]
    return mats


@njit()
def _create_ac_mats_no_filter(origin_ac, cmats):
    ac_mats = np.zeros_like(cmats)
    for i in range(len(cmats)):
        ac_mats[i] = origin_ac + cmats[i]
    return ac_mats

def _create_ac_matrices(origin_ac, cmats, ac_filters):
    ac_mats = _create_ac_mats_no_filter(origin_ac, cmats)
    criteria = _apply_filters_on_matrices(ac_mats, ac_filters)
    return _filter_by_boolian(criteria, ac_mats)
