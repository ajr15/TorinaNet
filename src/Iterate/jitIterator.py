import numpy as np
from itertools import combinations

from ..utils.TimeFunc import TimeFunc
from ..core.RxnGraph import RxnGraph
from ..core.Reaction import Reaction
from ..core.Specie import Specie
from ..core.AcMatrix.BinaryAcMatrix import BinaryAcMatrix
from .conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds, _TwoSpecieMatrix
from . import _jitted_commons as _jitted

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
    @TimeFunc
    def generate_single_bond_conversion_matrices(ac, conversion_filters, only_single_bonds):
        """Method to generate all single bond conversion matrices for a given ac matrix"""
        matrices = _jitted._generate_single_conversion_matrices_without_filter(ac, only_single_bonds)
        # applying criteria
        criteria = _jitted._apply_filters_on_matrices(matrices, conversion_filters)
        # filtering by criteria
        return _jitted._filter_by_boolian(criteria, matrices)

    @staticmethod
    @TimeFunc
    def gen_unique_idx_combinations_for_conv_mats(n_single_conv_mats, max_changing_bonds):
        combos = [np.array([i] + [0 for _ in range(max_changing_bonds - 1)]) for i in range(n_single_conv_mats)]
        for n in range(2, max_changing_bonds + 1):
            for c in combinations([j for j in range(n_single_conv_mats)], n):
                combos = combos + [np.array(list(c) + [0 for _ in range(max_changing_bonds - n)])]
        return np.array(combos)

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
        # JITing conversion matrix filters
        jitted_conversion_filters = _jitted.jit_conversion_filters(conversion_filters)
        # generating conversion matrices
        conv_matrices = cls.generate_single_bond_conversion_matrices(ac.matrix, jitted_conversion_filters, only_single_bonds)
        # if conv matrices is empty, return an empty list
        if len(conv_matrices) == 0:
            return conv_matrices
        n_conv_matrices = len(conv_matrices) # number of single bond transformations, used to add unique matrices in iteration step
        # iterate on conv_matrices (single bond changes) to get multi-bond change conversion matrices
        conv_idxs = cls.gen_unique_idx_combinations_for_conv_mats(n_conv_matrices, max_changing_bonds)
        multi_matrices = _jitted._make_multibond_conv_matrices(conv_idxs, conv_matrices)
        # making jitted versions of criteria
        # applying criteria
        criteria = _jitted._apply_filters_on_matrices(multi_matrices, jitted_conversion_filters)
        # filtering by criteria
        multi_matrices = _jitted._filter_by_boolian(criteria, multi_matrices)
        return np.concatenate((conv_matrices, multi_matrices))

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
        # cmats = self.generate_conversion_matrices(joint_ac, conversion_filters + [_TwoSpecieMatrix(max_l)])
        cmats = self.generate_conversion_matrices(joint_ac, conversion_filters)
        rxn_graph = RxnGraph() # init RxnGraph
        # adding original specie to the graph
        _specie1 = Specie.from_ac_matrix(ac1)
        _specie1.identifier = specie1.identifier
        rxn_graph.add_specie(_specie1)
        _specie2 = Specie.from_ac_matrix(ac2)
        _specie2.identifier = specie2.identifier
        rxn_graph.add_specie(_specie2)
        # apply all conversion matrices
        for new_ac in _jitted._create_ac_matrices(joint_ac.matrix, cmats, ac_filters):
            rxn_graph.add_reaction(Reaction.from_ac_matrices(joint_ac, self._ac_type(new_ac))) # adding rxn to graph makes sure its and its species are unique, 
                                                                                # no need to check here as well
                                                                                # TODO: properly separate reaction cases!
        return rxn_graph

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
        for new_ac in _jitted._create_ac_matrices(origin_ac.matrix, cmats, ac_filters):
            # decompose ac matrix to components
            rxn_graph.add_reaction(Reaction.from_ac_matrices(origin_ac, self._ac_type(new_ac))) # adding rxn to graph makes sure its and its species are unique, 
                                                                                 # as well as separating different reaction cases
                                                                                 # no need to check here as well!
        return rxn_graph

    def gererate_rxn_network(self, reactants: list, max_itr, ac_filters=[], conversion_filters=[]):
        """Function to iterate all possible elemetary reactions with given reactants"""
        # init
        rxn_graph = RxnGraph()
        for s in reactants:
            rxn_graph.add_specie(Specie.from_ac_matrix(self._ac_type.from_specie(s))) # properly add first specie to graph
        nspecies = rxn_graph.species
        # jitting ac_filters and conversion filters
        ac_filters = _jitted.jit_ac_filters(ac_filters)
        # iterate reactions of species
        counter = 1
        while True:
            nseed = RxnGraph()
            # add all unimolecular reactions
            print("*" * 10 + "  counter {}   ".format(counter) + "*" * 10)
            for s in nspecies:
                nseed.join(self.iterate_over_a_specie(s, ac_filters, conversion_filters))
            print("iterate_over_a_specie is ok")
            # add all bimolecular reactions
            # new species and old species
            for i in range(len(nspecies)):
                for j in range(len(rxn_graph.species)): 
                    nseed.join(self.iterate_over_species(nspecies[i], rxn_graph.species[j], ac_filters, conversion_filters))
            print("iterate_over_species is ok")
            # new species and new species
            for i in range(len(nspecies)):
                for j in range(1, len(nspecies)): 
                    nseed.join(self.iterate_over_species(nspecies[i], nspecies[j], ac_filters, conversion_filters))
            # get new species
            nspecies = nseed.compare_species(rxn_graph)
            # update rxn_graph
            rxn_graph.join(nseed)
            # check stop condition
            if max_itr <= counter: # TODO: implement proper stop condition!
                return rxn_graph
            counter += 1