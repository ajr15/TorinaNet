from .. import kernel
import sys


if not any(["iterate.kernels.numba" in k for k in sys.modules.keys()]):
    if kernel == "vanilla":
        from ..kernels import vanilla as utils
    elif kernel == "numba":
        from ..kernels import numba as utils
    else:
        raise ValueError("Unrecognized kernel name {}".format(kernel))

