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
        if all([not x > 1 and not x < -1 for v in matrix for x in v]):
            return True
        else:
            return False

class _TwoSpecieMatrix (ConvFilter):
    """Filter to make sure that only reactions between 2 species are created (to be used internaly only)"""

    def __init__(self, max_l):
        self.max_l = max_l
    
    def check(self, matrix: np.array):
        if np.sum(matrix[:self.max_l, self.max_l:]) > 0:
            return True
        else:
            return False
