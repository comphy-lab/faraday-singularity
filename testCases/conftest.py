"""The test suite's environment: MKL, OpenMP and OpenBLAS on one thread, whatever the caller set,
before pyoomph or numpy loads its libraries.

Bitwise identity of a restart with the run that wrote its state is a single-thread property: MKL's
threaded sparse direct solve (Pardiso) differs at round-off from one process to the next, and the
difference grows along a run (about 3e-7 relative in the momentum after two steps of the post-cut
continuation of ``test_ring_fem.py``). The tests that compare runs to the byte, or the pytest
process against the processes it starts, therefore need one thread in both: assigned here, it is
inherited by every process the tests start.
"""
import os

THREAD_VARIABLES = ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"
