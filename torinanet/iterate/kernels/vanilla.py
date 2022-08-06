"""Util functions for vanilla kernel (dask parallelization and no numba accelaration)"""
from ..filters.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds, _TwoSpecieMatrix
from .commons import gen_unique_idx_combinations_for_conv_mats
import numpy as np

def generate_single_bond_conversion_matrices(ac, conversion_filters, only_single_bonds):
    """Method to generate all single bond conversion matrices for a given ac matrix"""
    # remove the _TwoSpecieMatrix filter for the single conversion matrix formation
    filters = [c for c in conversion_filters if not type(c) is _TwoSpecieMatrix]
    # get all possible conv matrices (single bond change)
    mats = []
    for i in range(len(ac)):
        for j in range(i + 1, len(ac)):
            if ac[i][j] == 0: # makes sure there are no negative bond orders
                mat = np.zeros((len(ac), len(ac)))
                mat[i][j] = 1
                mat[j][i] = 1
                if all([c.check(mat) for c in filters]):
                    mats.append(mat)
            else: # makes sure only single bonds are created
                mat = np.zeros((len(ac), len(ac)))
                mat[i][j] = -1
                mat[j][i] = -1
                if all([c.check(mat) for c in filters]):
                    mats.append(mat)
                if not only_single_bonds:
                    mat = np.zeros((len(ac), len(ac)))
                    mat[i][j] = 1
                    mat[j][i] = 1
                    if all([c.check(mat) for c in filters]):
                        mats.append(mat)
    return mats


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
    conv_matrices = generate_single_bond_conversion_matrices(ac, conversion_filters, only_single_bonds)
    n_conv_matrices = len(conv_matrices) # number of single bond transformations, used to add unique matrices in iteration step
    # iterate on conv_matrices (single bond changes) to get multi-bond change conversion matrices
    for idxs in gen_unique_idx_combinations_for_conv_mats(n_conv_matrices, max_changing_bonds):
        mat = np.zeros((len(ac), len(ac)))
        for i in idxs:
            mat = mat + conv_matrices[i]
        if all([c.check(mat) for c in conversion_filters]):
            yield mat


def enumerate_over_ac(ac, conv_filters, ac_filters):
    ac_type = ac.__class__
    conv_mats = generate_conversion_matrices(ac.matrix, conv_filters)
    for conv_mat in conv_mats:
        new_ac = ac_type(ac.matrix.copy() + conv_mat)
        if all([f.check(new_ac) for f in ac_filters]):
            yield new_ac