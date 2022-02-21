"""Module to contain common util functions among all kernels"""
import numpy as np
from itertools import combinations
import sys
import threading
import time
from timeit import default_timer
from dask.callbacks import Callback


def join_matrices(mat1: np.ndarray, mat2: np.ndarray):
    """Method to make block matrix from two separate matrices. larger matrix is first."""
    l1 = len(mat1)
    l2 = len(mat2)
    if l2 > l1:
        return np.block([[mat2, np.zeros((l2, l1))], [np.zeros((l1, l2)), mat1]])
    else:
        return np.block([[mat1, np.zeros((l1, l2))], [np.zeros((l2, l1)), mat2]])


def gen_unique_idx_combinations_for_conv_mats(n_single_conv_mats, max_changing_bonds):
    for n in range(1, max_changing_bonds + 1):
        for c in combinations(range(n_single_conv_mats), n):
            yield list(c)




class ProgressCallback (Callback):
    """Progress callback to track progress of a dask computation

    Parameters
    ----------
    dt : float, optional
        Update resolution in seconds, default is 0.1 seconds
    out : file object, optional
        File object to which the progress bar will be written
        It can be ``sys.stdout``, ``sys.stderr`` or any other file object able to write ``str`` objects
        Default is ``sys.stdout``
    """

    def __init__(self, dt=0.1, out=None):
        if out is None:
            # Warning, on windows, stdout can still be None if
            # an application is started as GUI Application
            # https://docs.python.org/3/library/sys.html#sys.__stderr__
            out = sys.stdout
        self._dt = dt
        self._file = out
        self.ndone = 0

    def _start(self, dsk):
        self._state = None
        self._start_time = default_timer()
        # Start background thread
        self._running = True
        self._timer = threading.Thread(target=self._timer_func)
        self._timer.daemon = True
        self._timer.start()

    def _pretask(self, key, dsk, state):
        self._state = state
        if self._file is not None:
            self._file.flush()

    def _finish(self, dsk, state, errored):
        self._running = False
        self._timer.join()
        elapsed = default_timer() - self._start_time
        if self._file is not None:
            s = self._state
            ntasks = sum(len(s[k]) for k in ["ready", "waiting", "running", "finished"])
            self._write_msg(ntasks)
            self._file.write("done successfully in {} seconds\n".format(round(elapsed, 2)))
            self._file.flush()

    def _timer_func(self):
        """Background thread for updating the callback"""
        while self._running:
            if not self._state:
                ndone = 0
            else:
                ndone = len(self._state["finished"])
            if ndone > self.ndone:
                self.ndone = ndone
                self._write_msg(ndone)
            time.sleep(self._dt)

    def _write_msg(self, ndone):
        s = self._state
        if not s:
            return
        ntasks = sum(len(s[k]) for k in ["ready", "waiting", "running", "finished"])
        msg = "{} out of {} ({}%). total elapsed time = {} seconds\n".format(ndone, 
                                                                                ntasks, 
                                                                                round(self.ndone / ntasks * 100, 2),
                                                                                round(default_timer() - self._start_time, 2))
        self._file.write(msg)
        self._file.flush()
