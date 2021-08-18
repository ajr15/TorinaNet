import sys; sys.path.append("../..")
import numpy as np
from src.Iterate.conversion_matrix_filters import MaxChangingBonds, OnlySingleBonds, _TwoSpecieMatrix
from src.Iterate._jitted_commons import jit_conversion_filters, _apply_filters_on_matrices

if __name__ == '__main__':
    f = jit_conversion_filters([OnlySingleBonds(), _TwoSpecieMatrix(6)])
    data = np.random.randint(0, 2, size=(100, 10, 10))
    print(data)
    res = _apply_filters_on_matrices(data, f)
    print(res)
