from networkx.algorithms.assortativity.correlation import numeric_assortativity_coefficient
from networkx.algorithms.planarity import check_planarity
import numpy as np
from copy import deepcopy
from itertools import combinations
import dask as da
from dask.diagnostics import ProgressBar

from ..utils.TimeFunc import TimeFunc
from ..core.RxnGraph import RxnGraph
from ..core.Reaction import Reaction
from ..core.Specie import Specie
from ..core.AcMatrix.BinaryAcMatrix import BinaryAcMatrix
from .conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds, _TwoSpecieMatrix

class Iterator:
    """Object for handling all the elementary reaction iteration"""

    def __init__(self, ac_matrix_type=BinaryAcMatrix):
        self._ac_type = BinaryAcMatrix

    def join_ac_matrices(self, ac_matrix1, ac_matrix2):
        # make joint ac matrix with larger ac matrix first (top)
        # creating block matrix
        l1 = len(ac_matrix1)
        l2 = len(ac_matrix2)
        if l2 > l1:
            m = np.block([[ac_matrix2.matrix, np.zeros((l2, l1))], [np.zeros((l1, l2)), ac_matrix1.matrix]])
        else:
            m = np.block([[ac_matrix1.matrix, np.zeros((l1, l2))], [np.zeros((l2, l1)), ac_matrix2.matrix]])
        return self._ac_type(m)

    @staticmethod
    def generate_single_bond_conversion_matrices(ac, conversion_filters, max_changing_bonds, only_single_bonds):
        """Method to generate all single bond conversion matrices for a given ac matrix"""
        #TODO: WRITE WITH FIXED ARRAY SIZES !!!!!
        # get all possible conv matrices (single bond change)
        conv_matrices = []
        for i in range(len(ac)):
            for j in range(i + 1, len(ac)):
                if ac.matrix[i][j] == 0: # makes sure there are no negative bond orders
                    mat = np.zeros((len(ac), len(ac)))
                    mat[i][j] = 1
                    mat[j][i] = 1
                    if all([c.check(mat) for c in conversion_filters]):
                        conv_matrices.append(mat)
                else: # makes sure only single bonds are created
                    mat = np.zeros((len(ac), len(ac)))
                    mat[i][j] = -1
                    mat[j][i] = -1
                    if all([c.check(mat) for c in conversion_filters]):
                        conv_matrices.append(mat)
                    if not only_single_bonds:
                        mat = np.zeros((len(ac), len(ac)))
                        mat[i][j] = 1
                        mat[j][i] = 1
                        if all([c.check(mat) for c in conversion_filters]):
                            conv_matrices.append(mat)
        return conv_matrices


    @staticmethod
    def gen_unique_idx_combinations_for_conv_mats(n_single_conv_mats, max_changing_bonds):
        _idxs = [[i] for i in range(n_single_conv_mats)]
        combos = [] + _idxs
        for n in range(2, max_changing_bonds + 1):
            for c in combinations(range(n_single_conv_mats), n):
                combos += [list(c)]
        return combos

    @classmethod
    @TimeFunc
    def generate_conversion_matrices(cls, ac, conversion_filters):
        # TODO: possibly an iterator to save in RAM
        # TODO: try to implement it in fixed array sizes (no appends) for faster parallelization/execution
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
        conv_matrices = cls.generate_single_bond_conversion_matrices(ac, conversion_filters, max_changing_bonds, only_single_bonds)
        n_conv_matrices = len(conv_matrices) # number of single bond transformations, used to add unique matrices in iteration step
        # iterate on conv_matrices (single bond changes) to get multi-bond change conversion matrices
        for idxs in cls.gen_unique_idx_combinations_for_conv_mats(n_conv_matrices, max_changing_bonds):
            mat = np.zeros((len(ac), len(ac)))
            for i in idxs:
                mat = mat + conv_matrices[i]
            if all([c.check(mat) for c in conversion_filters]):
                conv_matrices.append(mat)
        return conv_matrices

    @da.delayed(pure=False)
    @TimeFunc
    def iterate_over_species(self, specie1, specie2, ac_filters, conversion_filters):
        """iterate over all possible reactions for 2 species"""
        # create joint ac matrix
        ac1 = self._ac_type.from_specie(specie1)
        ac2 = self._ac_type.from_specie(specie2)
        joint_ac = self.join_ac_matrices(ac1, ac2)
        # create all conversion matrices
        # make sure the conv matrices are indeed two species reactions
        # TODO: check for more efficient algorithm for that...
        # TODO: compare performance of filtering conversion matrices vs filtering reactions in graph
        max_l = max(len(ac1), len(ac2))
        cmats = self.generate_conversion_matrices(joint_ac, conversion_filters + [_TwoSpecieMatrix(max_l)])
        rxn_graph = RxnGraph() # init RxnGraph
        # adding original specie to the graph
        _specie1 = Specie.from_ac_matrix(ac1)
        _specie1.identifier = specie1.identifier
        rxn_graph.add_specie(_specie1)
        _specie2 = Specie.from_ac_matrix(ac2)
        _specie2.identifier = specie2.identifier
        rxn_graph.add_specie(_specie2)
        # apply all conversion matrices
        for cmat in cmats:
            # get new ac matrix
            new_ac = joint_ac.matrix + cmat
            new_ac = self._ac_type(new_ac.copy())
            # apply filters on ac matrix 
            if all([f(new_ac) for f in ac_filters]):
                # decompose ac matrix to components
                rxn_graph.add_reaction(Reaction.from_ac_matrices(joint_ac, new_ac)) # adding rxn to graph makes sure its and its species are unique, 
                                                                                    # no need to check here as well
                                                                                    # TODO: properly separate reaction cases!
        return rxn_graph

    @da.delayed(pure=False)
    @TimeFunc
    def iterate_over_a_specie(self, specie, ac_filters, conversion_filters):
        """iterate over all possible reactions for 2 species"""
        # create joint ac matrix
        origin_ac = self._ac_type.from_specie(specie)
        # create all conversion matrices
        cmats = self.generate_conversion_matrices(origin_ac, conversion_filters)
        rxn_graph = RxnGraph() # return a RxnGraph
        # adding original specie to the graph
        _specie = Specie.from_ac_matrix(origin_ac)
        _specie.identifier = specie.identifier
        rxn_graph.add_specie(_specie)
        # apply all conversion matrices
        for cmat in cmats:
            # get new ac matrix
            new_ac = origin_ac.matrix + cmat
            new_ac = self._ac_type(new_ac.copy())
            # apply filters on ac matrix 
            if all([f(new_ac) for f in ac_filters]):
                # decompose ac matrix to components
                rxn_graph.add_reaction(Reaction.from_ac_matrices(origin_ac, new_ac)) # adding rxn to graph makes sure its and its species are unique, 
                                                                                     # as well as separating different reaction cases
                                                                                     # no need to check here as well!
        return rxn_graph

    def gererate_rxn_network(self, reactants: list, max_itr, ac_filters=[], conversion_filters=[], verbose=1):
        """Function to iterate all possible elemetary reactions with given reactants"""
        # init
        rxn_graph = RxnGraph()
        reactant_species = []
        for s in reactants:
            specie = Specie.from_ac_matrix(self._ac_type.from_specie(s))
            rxn_graph.add_specie(specie) # properly add first specie to graph
            reactant_species.append(specie)
        rxn_graph.set_reactant_species(reactant_species)
        nspecies = rxn_graph.species
        # iterate reactions of species
        counter = 1
        while True:
            nseed = RxnGraph()
            # add all unimolecular reactions
            if verbose > 0:
                print("=" * 30)
                print(" " * 9 + "Iteration {}".format(counter))
                print("=" * 30)
                print("\nIterating over single specie reactions...")
            ajr = []
            for s in nspecies:
                ajr.append(self.iterate_over_a_specie(s, ac_filters, conversion_filters))
            if verbose > 0:
                with ProgressBar():
                    ajr = da.compute(ajr, scheduler="processes")[0]
                print("\nIterating over 2 specie reactions...")
            else:
                ajr = da.compute(ajr, scheduler="processes")[0]
            for ajr_rxn_graph in ajr:
                nseed.join(ajr_rxn_graph)
            # add all bimolecular reactions
            # new species and old species
            ajr = []
            for i in range(len(nspecies)):
                for j in range(len(rxn_graph.species)): 
                    ajr.append(self.iterate_over_species(nspecies[i], rxn_graph.species[j], ac_filters, conversion_filters))
            if verbose > 0:
                with ProgressBar():
                    ajr = da.compute(ajr, scheduler="processes")[0]
            else:
                ajr = da.compute(ajr, scheduler="processes")[0]
            for ajr_rxn_graph in ajr:
                nseed.join(ajr_rxn_graph)
            # new species and new species
            ajr = []
            for i in range(len(nspecies)):
                for j in range(i, len(nspecies)): 
                    ajr.append(self.iterate_over_species(nspecies[i], nspecies[j], ac_filters, conversion_filters))
            if verbose > 0:
                with ProgressBar():
                    ajr = da.compute(ajr, scheduler="processes")[0]
            else:
                ajr = da.compute(ajr, scheduler="processes")[0]
            for ajr_rxn_graph in ajr:
                nseed.join(ajr_rxn_graph)
            if verbose > 0:
                print("DONE ITERATING REACTIONS IN THIS ITERATION SUCCESSFULLY")
            # get new species
            nspecies = nseed.compare_species(rxn_graph)
            # update rxn_graph
            rxn_graph.join(nseed)
            # check stop condition
            if max_itr <= counter: # TODO: implement proper stop condition!
                if verbose > 0:
                    print("Stopping condition met!")
                    print("Finished reaction iteration successfully")
                return rxn_graph
            if verbose > 0:
                print("Stopping condition is not met, continueing...")
            counter += 1