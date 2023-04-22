from abc import ABC, abstractclassmethod
import numpy as np

class ConvFilter (ABC):

    def __init__(self):
        raise NotImplementedError
    
    @abstractclassmethod
    def check(self, matrix: np.array):
        """Method to check if the filter's criterion is met.
        ARGS:
            - matrix (np.array): conversion matrix
        RETURNS:
            (bool) criterion is met or not"""
        raise NotImplementedError

class MaxChangingBonds (ConvFilter):
    """Set maximum number of bonds to change. MUST specify for all computations
    ARGS:
        - n (int): number of maxinum changing bonds"""

    def __init__(self, n):
        self.n = n
    
    def check(self, matrix: np.array):
        return True

class OnlySingleBonds (ConvFilter):
    """Filter to make conversion matrix creat bonds of order no greater than single (mainly used for binary ac matrices)"""

    def __init__(self):
        pass

    def check(self, matrix: np.array):
        # if all([not x > 1 and not x < -1 for v in matrix for x in v]):
        #     return True
        # else:
        #     return False
        return True
    
class MaxFormingAndBreakingBonds (ConvFilter):
    """Filter to limit total number of bonds breaking or forming per conversion matrix"""

    def __init__(self, max_forming: int=-1, max_breaking: int=-1):
        self.max_forming = max_forming
        self.max_breaking = max_breaking

    def check(self, matrix: np.array):
        # calculates number of forming bonds, by canceling out the breaking bonds
        if self.max_forming != -1:
            n_forming = int(np.sum(np.abs(matrix) + matrix) / 2)
        else:
            n_forming = -1
        # similarly calculates number of breaking bonds
        if self.max_breaking != -1:
            n_breaking = - int(np.sum(np.abs(matrix) - matrix) / 2)
        else:
            n_breaking = -1
        return n_forming <= self.max_forming and n_breaking <= self.max_breaking

class _TwoSpecieMatrix (ConvFilter):
    """Filter to make sure that only reactions between 2 species are created (to be used internaly only)"""

    def __init__(self, max_l):
        self.max_l = max_l
    
    def check(self, matrix: np.array):
        if np.sum(matrix[:self.max_l, self.max_l:]) > 0:
            return True
        else:
            return False
