cimport mpi4py.MPI as MPI

# See https://bitbucket.org/mpi4py/mpi4py/issues/1/mpi4py-cython-openmpi
cdef extern from 'mpi-compat.h': pass

# Make sure these definitions match those in "picksc/ppic2/precision.h"
ctypedef double real_t
ctypedef double complex complex_t

cdef struct particle_t:
    real_t x, y, vx, vy

cdef struct real2_t:
    real_t x, y

cdef struct complex2_t:
    complex_t x, y

cdef class grid_t:
    """Grid extension type.
    This is inherited by the Grid class (see grid.py)."""
    # TODO: Either define *all* attributes of the grid class or only those that
    # are actually needed
    cdef public int nx, ny
    cdef public MPI.Comm comm
    cdef public real_t edges[2]
    cdef public int nyp, noff, nypmx, nypmn
    cdef public int lbx, lby
